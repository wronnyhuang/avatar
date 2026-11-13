"""multi.py with a fake trainer: concurrency, batches, per-run env, deadline budget + skip, SIGTERM across batches.

    python -I -m pytest -q tools/train/test_multi.py
"""
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = r'''
import argparse, json, os, signal, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--config"); ap.add_argument("--run-dir"); ap.add_argument("--pack-dir")
a = ap.parse_args()
cfg = json.load(open(a.config))
def term(s, f):
    open(os.path.join(a.run_dir, "TERMED"), "w").write("1")
    sys.exit(143)
signal.signal(signal.SIGTERM, term)
info = dict(t0=time.time(), budget=os.environ.get("TRAIN_TIME_BUDGET_S"), flavor=os.environ.get("FLAVOR"),
            common=os.environ.get("COMMON_ENV"), cfg=cfg)
open(os.path.join(a.run_dir, "started.json"), "w").write(json.dumps(info))
time.sleep(float(cfg.get("sleep", 0.5)))
info["t1"] = time.time()
json.dump(info, open(os.path.join(a.run_dir, "DONE"), "w"))
sys.exit(int(cfg.get("rc", 0)))
'''


def _setup(tmp_path, mc):
    fake = tmp_path / "fake_train.py"
    fake.write_text(FAKE)
    cp = tmp_path / "multi.json"
    cp.write_text(json.dumps(mc))
    rd = tmp_path / "R"
    cmd = [sys.executable, "-I", os.path.join(HERE, "multi.py"), "--config", str(cp), "--run-dir", str(rd),
           "--pack-dir", str(tmp_path), "--train-script", str(fake), "--stagger", "0"]
    return cmd, rd


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if k not in ("TRAIN_DEADLINE_UNIX", "TRAIN_TIME_BUDGET_S")}
    env.update({k: str(v) for k, v in kw.items()})
    return env


def _done(rd):
    return json.loads((rd / "DONE").read_text())


def test_all_at_once_without_batches(tmp_path):
    cmd, rd = _setup(tmp_path, {"common": {"sleep": 1.0, "x": {"a": 1}}, "runs": {"a": {}, "b": {"x": {"b": 2}, "rc": 3}}})
    subprocess.run(cmd, check=True, env=_env(), timeout=60)
    d = _done(rd)
    assert d["skipped"] == {} and d["a"]["rc"] == 0 and d["b"]["rc"] == 3
    a, b = d["a"]["done"], d["b"]["done"]
    assert max(a["t0"], b["t0"]) < min(a["t1"], b["t1"])           # overlapped
    assert b["cfg"]["x"] == {"a": 1, "b": 2} and a["budget"] is None
    assert "env" not in a["cfg"]


def test_batches_run_in_sequence_with_env_and_budget(tmp_path):
    mc = {"common": {"sleep": 1.0, "env": {"COMMON_ENV": "c", "FLAVOR": "base"}},
          "runs": {"a": {"env": {"FLAVOR": "A"}}, "b": {}, "c": {"env": {"FLAVOR": "C"}}, "d": {}},
          "batches": [["a", "b"], ["c"]]}
    cmd, rd = _setup(tmp_path, mc)
    dl = time.time() + 3600
    subprocess.run(cmd, check=True, env=_env(TRAIN_DEADLINE_UNIX=int(dl)), timeout=60)
    d = _done(rd)
    assert d["skipped"] == {} and all(d[n]["rc"] == 0 for n in "abcd")
    a, b, c, dd = (d[n]["done"] for n in "abcd")
    assert max(a["t0"], b["t0"]) < min(a["t1"], b["t1"])            # inside a batch: concurrent
    assert c["t0"] >= max(a["t1"], b["t1"])                         # batch 2 after batch 1 exited
    assert dd["t0"] >= c["t1"]                                      # unlisted run -> a final batch
    assert (a["flavor"], b["flavor"], c["flavor"]) == ("A", "base", "C") and a["common"] == "c"
    assert 3500 < int(a["budget"]) <= 3600 and int(c["budget"]) <= int(a["budget"]) - 1
    assert abs(int(c["budget"]) - (dl - c["t0"])) < 5               # recomputed from the deadline


def test_batch_skipped_when_too_little_time_left(tmp_path):
    mc = {"common": {"sleep": 3.0}, "runs": {"a": {}, "b": {}, "c": {}}, "batches": [["a"], ["b", "c"]],
          "min_batch_minutes": 0.1}                                  # 6 s
    cmd, rd = _setup(tmp_path, mc)
    t = time.time()
    out = subprocess.run(cmd, check=True, env=_env(TRAIN_TIME_BUDGET_S=8), timeout=60, capture_output=True, text=True)
    d = _done(rd)
    assert d["a"]["rc"] == 0 and 0 < int(d["a"]["done"]["budget"]) <= 8
    assert set(d["skipped"]) == {"b", "c"} and "min_batch_minutes" in d["skipped"]["b"]
    assert "b" not in d and not (rd / "b").exists() and "skipped" in out.stdout
    assert time.time() - t < 30


def test_sigterm_forwarded_and_no_new_batch(tmp_path):
    mc = {"common": {"sleep": 60}, "runs": {"a": {}, "b": {}, "c": {}}, "batches": [["a", "b"], ["c"]]}
    cmd, rd = _setup(tmp_path, mc)
    p = subprocess.Popen(cmd, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t = time.time()
    while not ((rd / "a" / "started.json").exists() and (rd / "b" / "started.json").exists()):
        assert time.time() - t < 60 and p.poll() is None
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)
    d = _done(rd)
    assert d["a"]["rc"] == 143 and d["b"]["rc"] == 143
    assert (rd / "a" / "TERMED").exists() and (rd / "b" / "TERMED").exists()
    assert d["skipped"] == {"c": "sigterm"} and not (rd / "c").exists()
    assert time.time() - t < 60


def test_data_wait_ready_marker_and_skip(tmp_path):
    (tmp_path / "present.npz").write_text("x")                     # present before the start: counts as there
    mc = {"common": {"sleep": 0.2}, "max_wait_minutes": 0.15,
          "runs": {"a": {"pack_extra": ["present.npz"]}, "b": {"pack_extra": ["late.npz"]}, "c": {"needs": ["never.npz"]}},
          "batches": [["a"], ["b", "c"]]}
    cmd, rd = _setup(tmp_path, mc)
    p = subprocess.Popen(cmd, env=_env())
    time.sleep(1.5)
    (tmp_path / "late.npz").write_text("x")                        # pushed mid-job: needs its .ready marker
    time.sleep(1.0)
    (tmp_path / "late.npz.ready").write_text("")
    assert p.wait(timeout=60) == 0
    d = _done(rd)
    assert d["a"]["rc"] == 0 and d["b"]["rc"] == 0
    assert "c" in d["skipped"] and "never.npz" in d["skipped"]["c"]
