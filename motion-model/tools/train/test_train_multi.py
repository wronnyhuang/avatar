"""train_multi.py (several models in lockstep on one Sampler) on a fake v3 pack (synth.py -> qpack -> fake_v3 members).

    nice -n 19 python -I -m pytest -q tools/train/test_train_multi.py

1. one model in a lane == train.py (same rows, same bundle weights); 2. flow_loss never mutates the shared batch;
3. isolate_eval_rng: evals no longer restart the training batch stream; 4. a 3-model lane (sizes, lrs, a patience
stop that frees its model while the others go on) + the shared-key check; 5. through multi.py; 6. SIGTERM -> every
model has a bundle from its BEST EMA + DONE interrupted, exit 0; 7. a lane bundle passes unified/export.py and
unified/test_runtime.py (as tools/train/p2_collect.py runs them).
"""
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import fake_v3  # noqa: E402
import train as T1  # noqa: E402
from unified.data import Sampler  # noqa: E402
from unified.flow import flow_loss  # noqa: E402

ENC = "whisper_tiny"
TINY = {"d": 32, "layers": 1, "heads": 2, "hd": 32, "hlayers": 1, "hheads": 2}
BASE = dict(pack="v3.npz", model=TINY, batch=4, seq=60, chunks=2, steps=20, warmup=5, eval_every=10, log_every=5,
            val_batches=1, threads=3, max_minutes=1e6, style_procs=2, events=True, audio_in="enc_" + ENC,
            loss={"a2f": 0.1})
TRAIN, MULTI, LANE = (os.path.join(HERE, f) for f in ("train.py", "train_multi.py", "multi.py"))


def _env():
    env = {k: v for k, v in os.environ.items() if k not in ("TRAIN_DEADLINE_UNIX", "TRAIN_TIME_BUDGET_S")}
    env.update(OMP_NUM_THREADS="3", MKL_NUM_THREADS="3")
    return env


def _cmd(script, cfg_path, rd, pack_dir):
    return [sys.executable, "-I", script, "--config", str(cfg_path), "--run-dir", str(rd), "--pack-dir", str(pack_dir)]


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return p


def _rows(rd):
    return [json.loads(x) for x in open(os.path.join(rd, "metrics.jsonl"))]


def _json(p):
    with open(p) as f:
        return json.load(f)


def _close(a, b, tol=1e-6):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a == b or abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    d = tmp_path_factory.mktemp("mt")
    q = fake_v3.synth_qpack(str(d / "syn"), sessions=6, minutes=1.0)
    fake_v3.add_v3_members(q, str(d / "v3.npz"), {ENC: 48})
    return d


@pytest.fixture(scope="module")
def smp(pack):
    torch.set_num_threads(3)
    cfg = dict(T1.DEFAULTS, **copy.deepcopy(BASE))
    s = Sampler(str(pack / "v3.npz"), "cpu", seed=0, **T1.sampler_kwargs(cfg, str(pack)))
    T1.complete_model_cfg(cfg, s)
    return s, cfg


def test_one_model_lane_equals_train_py(pack, tmp_path):
    c1 = _write(tmp_path / "one.json", BASE)
    cm = _write(tmp_path / "lane.json", dict(BASE, models={"a": {}}))
    r1, rm = tmp_path / "r_train", tmp_path / "r_multi"
    p1 = subprocess.Popen(_cmd(TRAIN, c1, r1, pack), env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pm = subprocess.Popen(_cmd(MULTI, cm, rm, pack), env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    o1, om = p1.communicate(timeout=600)[0].decode(), pm.communicate(timeout=600)[0].decode()
    assert p1.returncode == 0, o1[-3000:]
    assert pm.returncode == 0, om[-3000:]
    ra = [{k: v for k, v in r.items() if k != "t"} for r in _rows(r1)]
    rb = [{k: v for k, v in r.items() if k != "t"} for r in _rows(rm / "a")]
    assert len(ra) == len(rb) == 8                                  # start, 4 log, 2 val, final
    assert [r.get("step") for r in ra] == [r.get("step") for r in rb]
    for x, y in zip(ra, rb):
        assert _close(x, y), (x, y)
    assert any("val_core" in r and not r.get("final") for r in rb) and rb[-1]["final"] and "val_drop" in rb[-1]
    wa, wb = torch.load(r1 / "bundle" / "weights.pt"), torch.load(rm / "a" / "bundle" / "weights.pt")
    assert wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
    ca, cb = _json(r1 / "bundle" / "config.json"), _json(rm / "a" / "bundle" / "config.json")
    assert cb.pop("lane") == "r_multi" and cb.pop("model_name") == "a"
    for c in (ca, cb):
        c.pop("train_minutes"); c.pop("final_val")
    assert ca == cb
    da, db = _json(r1 / "DONE"), _json(rm / "a" / "DONE")
    assert set(da) <= set(db) and db["lane"] == "r_multi" and db["model"] == "a" and db["interrupted"] is False
    assert _close(da["final_val"], db["final_val"]) and db["final_phase_s"] > 0
    assert not (rm / "a" / "ckpt_final.pt").exists() and (r1 / "ckpt_final.pt").exists()
    assert "model" not in torch.load(rm / "a" / "ckpt_best.pt") and "model" in torch.load(r1 / "ckpt_best.pt")
    lane = _json(rm / "DONE")
    assert set(lane) == {"a"} and lane["a"]["step"] == 20 and lane["a"]["interrupted"] is False


def test_flow_loss_does_not_mutate_the_batch(smp):
    s, cfg = smp
    torch.manual_seed(1)
    A, _, _ = T1.build_model(cfg, "cpu")
    B, _, optB = T1.build_model(dict(cfg, model=dict(cfg["model"], d=48, hd=48)), "cpu")
    b = s.batch(4, 60, 2, split="train", aug=cfg["aug"])
    b0 = {k: v.clone() for k, v in b.items()}
    with torch.no_grad():
        torch.manual_seed(7); la1, pa1 = flow_loss(A, b, s.colw, cfg["loss"])
    torch.manual_seed(8)
    lb, _ = flow_loss(B, b, s.colw, cfg["loss"])                    # a training step of B on the same batch
    optB.zero_grad(); lb.backward(); optB.step()
    with torch.no_grad():
        torch.manual_seed(7); la2, pa2 = flow_loss(A, b, s.colw, cfg["loss"])
    assert torch.equal(la1, la2) and all(torch.equal(pa1[k], pa2[k]) for k in pa1)
    assert b.keys() == b0.keys() and all(torch.equal(b[k], b0[k]) for k in b)


def test_isolate_eval_rng(smp):
    s, cfg = smp
    m, _, _ = T1.build_model(cfg, "cpu")
    picks = {}
    for iso in (False, True):
        c = dict(cfg, isolate_eval_rng=iso)
        s.g.manual_seed(123)
        seq = []
        for _ in range(2):                                         # eval, then 2 training batches, twice
            g0, r0 = s.g.get_state(), torch.get_rng_state()
            T1.val(m, s, c, n_batches=1, sample=False)
            if iso:
                assert torch.equal(s.g.get_state(), g0) and torch.equal(torch.get_rng_state(), r0)
            seq.append(torch.cat([s.batch(4, 60, 2, split="train")["s0"] for _ in range(2)]))
        picks[iso] = seq
    assert torch.equal(picks[False][0], picks[False][1])            # train.py default: the post-eval batches repeat
    assert not torch.equal(picks[True][0], picks[True][1])


@pytest.fixture(scope="module")
def lane3(pack, tmp_path_factory):
    d = tmp_path_factory.mktemp("lane3")
    cfg = dict(BASE, steps=30, isolate_eval_rng=True, final_drops=False,
               models={"a": {"sweep": {"lane": "d25", "size": "a", "frac": 0.25}},
                       "b": {"model": {"d": 48, "layers": 2, "hd": 48}, "lr": 1e-3, "warmup": 4,
                             "sweep": {"lane": "d25", "size": "b", "frac": 0.25}},
                       "c": {"lr": 0.0, "patience_evals": 1, "sweep": {"lane": "d25", "size": "c", "frac": 0.25}}})
    rd = d / "d25"
    out = subprocess.run(_cmd(MULTI, _write(d / "lane.json", cfg), rd, pack), env=_env(), capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, (out.stdout[-3000:], out.stderr[-3000:])
    return rd, out.stdout


def test_lane_of_three(lane3):
    rd, stdout = lane3
    lane = _json(rd / "DONE")
    assert set(lane) == {"a", "b", "c"}
    params = {}
    for n in "abc":
        md = rd / n
        rows = _rows(md)
        assert rows[0]["event"] == "start" and rows[0]["params"] > 0
        params[n] = rows[0]["params"]
        assert [r["step"] for r in rows if "val_core" in r and not r.get("final")] == ([10, 20] if n == "c" else [10, 20, 30])
        assert rows[-1]["final"] and "events_cal" in rows[-1] and "val_drop" not in rows[-1]
        c = _json(md / "cfg.json")
        assert c["sweep"]["size"] == n and "models" not in c and c["model"]["layers"] == (2 if n == "b" else 1)
        bc = _json(md / "bundle" / "config.json")
        assert bc["model"]["d"] == (48 if n == "b" else 32) and bc["train_config"]["sweep"]["size"] == n
        assert (md / "bundle" / "weights.pt").exists() and (md / "ckpt_last.pt").exists()
        d = _json(md / "DONE")
        assert d["model"] == n and d["lane"] == "d25" and d["sweep"]["frac"] == 0.25 and d["interrupted"] is False
        assert lane[n]["step"] == d["step"] and lane[n]["best_step"] == d["best_step"]
    assert params["b"] > params["a"] == params["c"]
    assert lane["a"]["step"] == lane["b"]["step"] == 30 and lane["c"]["step"] == 20
    assert lane["c"]["stop_reason"].startswith("patience: no val_core improvement in 1 evals")
    assert lane["c"]["best_step"] == 10 and lane["a"]["stop_reason"] == "steps done"
    lr_b = [r["lr"] for r in _rows(rd / "b") if "lr" in r]
    lr_a = [r["lr"] for r in _rows(rd / "a") if "lr" in r]
    assert lr_b[0] != lr_a[0] and all(r["lr"] == 0 for r in _rows(rd / "c") if "lr" in r)
    # c finished (and left the active set) while a and b went on training
    lines = [json.loads(x) for x in stdout.splitlines() if x.startswith("{")]
    i_c = next(i for i, x in enumerate(lines) if x.get("event") == "model_done" and x["model"] == "c")
    assert lines[i_c + 1] == {"event": "active", "models": ["a", "b"]}
    assert any(x.get("model") == "a" and x.get("step", 0) > 20 and "lr" in x for x in lines[i_c:])
    assert any(x.get("model") == "b" and x.get("step", 0) > 20 and "lr" in x for x in lines[i_c:])


def test_shared_key_mismatch_is_refused(pack, tmp_path):
    cfg = dict(BASE, models={"a": {}, "b": {"batch": 8, "aug": {"p_mirror": 0.0}}, "c": {"model": {"horizon": 12}}})
    out = subprocess.run(_cmd(MULTI, _write(tmp_path / "bad.json", cfg), tmp_path / "R", pack), env=_env(),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode != 0
    msg = out.stderr
    assert "must be the same for every model" in msg and "batch: a=4, b=8, c=4" in msg
    assert "aug:" in msg and "model.horizon: a=15, b=15, c=12" in msg
    assert not (tmp_path / "R" / "a" / "metrics.jsonl").exists()
    out = subprocess.run(_cmd(MULTI, _write(tmp_path / "none.json", BASE), tmp_path / "R2", pack), env=_env(),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and '"models"' in out.stderr


def test_through_multi_py(pack, tmp_path):
    mc = {"common": dict(BASE, steps=10, eval_every=5, final_drops=False),
          "runs": {"d25": {"models": {"s1": {"sweep": {"lane": "d25", "size": "s1"}},
                                      "s2": {"model": {"layers": 2}, "lr": 1e-3, "sweep": {"lane": "d25", "size": "s2"}}},
                           "env": {"LANE_FLAVOR": "x"}}}}
    cmd = [sys.executable, "-I", LANE, "--config", str(_write(tmp_path / "m.json", mc)), "--run-dir",
           str(tmp_path / "R"), "--pack-dir", str(pack), "--train-script", MULTI, "--stagger", "0"]
    out = subprocess.run(cmd, env=_env(), capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-2000:])
    R = tmp_path / "R"
    d = _json(R / "DONE")
    assert d["skipped"] == {} and d["d25"]["rc"] == 0
    assert set(d["d25"]["done"]) == {"s1", "s2"} and all(d["d25"]["done"][n]["step"] == 10 for n in ("s1", "s2"))
    for n in ("s1", "s2"):
        assert (R / "d25" / n / "bundle" / "weights.pt").exists() and (R / "d25" / n / "DONE").exists()
        assert _json(R / "d25" / n / "cfg.json")["sweep"]["size"] == n
    assert "env" not in _json(R / "d25" / "cfg.json")
    assert '"event": "lane_done"' in (R / "d25" / "stdout.log").read_text()


def test_sigterm_bundles_best_ema_and_exits_0(pack, tmp_path):
    cfg = dict(BASE, steps=100000, eval_every=5, models={"a": {}, "b": {"model": {"layers": 2}, "lr": 1e-3}})
    rd = tmp_path / "R"
    log = open(tmp_path / "out.log", "w")
    p = subprocess.Popen(_cmd(MULTI, _write(tmp_path / "lane.json", cfg), rd, pack), env=_env(), stdout=log,
                         stderr=subprocess.STDOUT)
    t = time.time()

    def evaluated(n):
        mp = rd / n / "metrics.jsonl"
        return mp.exists() and any("val_core" in json.loads(x) for x in open(mp) if x.strip())
    while not (evaluated("a") and evaluated("b")):
        assert p.poll() is None and time.time() - t < 300, (tmp_path / "out.log").read_text()[-3000:]
        time.sleep(0.2)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=300) == 0, (tmp_path / "out.log").read_text()[-3000:]
    log.close()
    lane = _json(rd / "DONE")
    for n in ("a", "b"):
        md = rd / n
        d = _json(md / "DONE")
        assert d["interrupted"] is True and d["final_val"]["stop_reason"] == "sigterm" and lane[n]["interrupted"] is True
        assert 5 <= d["step"] < 100000 and lane[n]["step"] == d["step"]
        best = torch.load(md / "ckpt_best.pt")
        w = torch.load(md / "bundle" / "weights.pt")
        assert set(best) == {"ema", "step", "val", "cfg"}
        assert w.keys() == best["ema"].keys() and all(torch.equal(w[k], best["ema"][k]) for k in w)   # the BEST EMA
        assert d["final_val"]["from_best_step"] == best["step"] == d["best_step"]
        last = torch.load(md / "ckpt_last.pt")
        assert last["step"] == d["step"]
        fr = _rows(md)[-1]
        assert fr["final"] and fr["interrupted"] and "events_cal" in fr
        bc = _json(md / "bundle" / "config.json")
        assert bc["interrupted"] is True and bc["events"]["calibration"]["thresholds"]
    # relaunch = resume: both models continue from ckpt_last and finish normally
    s1 = {n: lane[n]["step"] for n in ("a", "b")}
    goal = max(s1.values()) + 6
    out = subprocess.run(_cmd(MULTI, tmp_path / "lane.json", rd, pack) + ["--set", f"steps={goal}", "--set",
                         "final_drops=false"], env=_env(), capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:]
    lane = _json(rd / "DONE")
    for n in ("a", "b"):
        assert lane[n]["step"] == goal and lane[n]["interrupted"] is False and lane[n]["stop_reason"] == "steps done"
        starts = [r for r in _rows(rd / n) if r.get("event") == "start"]
        assert [r["step"] for r in starts] == [0, s1[n]]
        assert f"{n}: resumed from step {s1[n]}" in out.stdout
    out = subprocess.run(_cmd(MULTI, tmp_path / "lane.json", rd, pack), env=_env(), capture_output=True, text=True,
                         timeout=120)                      # everything DONE: nothing is trained or loaded again
    assert out.returncode == 0 and "a: already DONE" in out.stdout and "b: already DONE" in out.stdout
    assert "events:" not in out.stdout and _json(rd / "DONE") == lane


def test_lane_bundle_exports_and_runs(lane3, tmp_path):
    rd, _ = lane3
    b = tmp_path / "bundle"
    shutil.copytree(rd / "b" / "bundle", b)
    env = _env()
    out = subprocess.run([sys.executable, "-I", "unified/export.py", str(b)], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-2000:])
    assert (b / "head_step.onnx").exists()
    out = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "unified/test_runtime.py"], cwd=ROOT,
                         env=dict(env, UNIFIED_BUNDLE=str(b)), capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-3000:]
