"""motion: the living-person layer (renderer/humans/human_life.py) that junor's face json switches on.

Runs junor's face table on the asset at hand (junor's when published, else Juno's: the life layer only reads the shared
skeleton / ARKit names).  No GL.   python -m pytest tests/test_junor_life.py -q
"""
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np
try:
    import pytest
except ImportError:      # the Quick Start venv has no pytest: nothing to run
    print("pytest not installed: test_junor_life skipped")
    raise SystemExit(0)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RR = os.path.join(ROOT, "renderer")
sys.path.insert(0, RR)
from anim.schema import IDX, NPARAMS  # noqa: E402
from humans import rig as RIG  # noqa: E402

JUNOR_FACE = os.path.join(RR, "humans", "junor", "junor.face.json")


def _asset():
    for p in (os.environ.get("JUNOR_TEST_ASSET", ""), os.path.join(RR, "humans", "junor", "junor.asset"),
              os.path.join(RR, "humans", "juno", "juno.asset")):
        if os.path.isfile(os.path.join(p, "skeleton.json")):
            return p
    pytest.skip("no human asset")


ASSET = _asset()
CAM = (0.0, 1.9, 6.0)


def junor(**kw):
    return RIG.JunoRig(ASSET, face_json=JUNOR_FACE, **kw)


def vec(d=None, **kw):
    P = np.zeros(NPARAMS)
    for k, v in dict(d or {}, **kw).items():
        P[IDX[k]] = v
    return P


@pytest.fixture(scope="module")
def r():
    return junor()


def test_life_is_on_for_junor_only(r):
    assert r.life is not None and all(r.life.on.values())
    j = RIG.JunoRig(os.path.join(RR, "humans", "juno", "juno.asset")) if os.path.isdir(os.path.join(RR, "humans", "juno", "juno.asset")) else None
    if j is not None:
        assert j.life is None


def test_face_json_follows_the_asset_folder():
    d = tempfile.mkdtemp(prefix="junor_asset_")
    a = os.path.join(d, "junor.asset")
    os.makedirs(a)
    for f in ("skeleton.json", "juno.asset.json"):
        shutil.copy(os.path.join(ASSET, f), a)
    os.environ.pop("JUNO_FACE_JSON", None)
    rg = RIG.JunoRig(a)
    assert rg.life is not None
    assert RIG._face_path(os.path.join(d, "juno.asset"), {}).endswith(os.path.join("juno", "juno.face.json"))


def _run(rg, n, fps, P=None, t0=0.0):
    out = []
    for i in range(n):
        t = t0 + i/fps
        f = rg.solve(P if P is not None else vec(), t=t, cam_pos=CAM)
        out.append((f, dict(rg._eye_ang)))
    return out


def test_deterministic_given_the_frame_sequence():
    a, b = junor(), junor()
    P = vec(smile=0.4, lip_open=0.3, teeth_gap=0.3, voice=0.7, breath=0.5, head_yaw=0.3, blink_l=0.6, blink_r=0.6)
    for i in range(30):
        fa = a.solve(P, t=i/24, cam_pos=CAM); fb = b.solve(P, t=i/24, cam_pos=CAM)
        assert np.array_equal(fa.bones, fb.bones) and np.array_equal(fa.weights, fb.weights)
    f1 = a.solve(P, t=29/24, cam_pos=CAM); f2 = a.solve(P, t=29/24, cam_pos=CAM)      # the same frame twice
    assert np.array_equal(f1.bones, f2.bones) and np.array_equal(f1.weights, f2.weights)


def test_fixation_scanning_rate_amplitude_and_main_sequence():
    rg = junor()
    lf = rg.life
    fps = 500.0
    ts = np.arange(0.0, 30.0, 1/fps)
    yp = np.degrees(np.array([lf.eye_offset(t, 0.0, 0.0, 0.0) for t in ts]))
    v = np.hypot(*np.diff(yp, axis=0).T)*fps                  # deg/s
    sac = v > 30.0
    starts = np.nonzero(sac[1:] & ~sac[:-1])[0]
    rate = len(starts)/30.0
    assert 1.0 <= rate <= 3.2, rate
    assert np.abs(yp).max() < 5.0
    # main sequence: peak velocity grows with amplitude, durations 20-40 ms for these 1-6 deg saccades
    ends = np.nonzero(~sac[1:] & sac[:-1])[0]
    durs = []
    for s in starts:
        e = ends[ends > s]
        if len(e):
            durs.append((e[0] - s)/fps)
    assert 0.012 < np.median(durs) < 0.045, np.median(durs)
    assert v.max() < 400.0


def test_vergence_converges_on_the_camera_distance(r):
    for gx in (0.0, 0.4, -0.4):
        f = r.solve(vec(gaze_x=gx), t=0.0, cam_pos=CAM)
        e = f.eyes.astype(float)
        bi = r.bi
        dl = f.bones[bi["eye_l"], :3, :3].astype(float) @ np.linalg.inv(r.Mrest_lin[bi["eye_l"]]) @ np.array([0, 0, 1.0])
        dr = f.bones[bi["eye_r"], :3, :3].astype(float) @ np.linalg.inv(r.Mrest_lin[bi["eye_r"]]) @ np.array([0, 0, 1.0])
        dl /= np.linalg.norm(dl); dr /= np.linalg.norm(dr)
        # never wall-eyed: the lines of sight approach each other in front of the face
        w = e[0] - e[1]
        a, b, c, d_, e_ = dl @ dl, dl @ dr, dr @ dr, dl @ w, dr @ w
        den = a*c - b*b
        if den > 1e-12:
            s = (b*e_ - c*d_)/den
            assert s > 0.5, (gx, s)                   # converge ahead, at a positive (large) distance
        ang = math.degrees(math.acos(np.clip(dl @ dr, -1, 1)))
        assert ang < 3.0, ang                          # a sane convergence angle at a conversational distance


def _w(rg, f, n):
    return float(f.weights[rg.si[n]]) if n in rg.si else 0.0


def test_blink_one_eye_lags_and_both_close_fully():
    rg = junor()
    seq = [0.0]*6 + [0.3, 0.8, 1.0, 1.0, 0.7, 0.35, 0.1] + [0.0]*6
    L, R, sq = [], [], []
    for i, b in enumerate(seq):
        f = rg.solve(vec(blink_l=b, blink_r=b), t=i/48, cam_pos=CAM)
        L.append(_w(rg, f, "eyeBlinkLeft")); R.append(_w(rg, f, "eyeBlinkRight")); sq.append(_w(rg, f, "eyeSquintLeft"))
    L, R = np.array(L), np.array(R)
    assert L.max() > 0.97 and R.max() > 0.97
    assert (L[6:8] < R[6:8] - 0.01).all()             # the lag eye (anatomical left) is behind on the close
    assert max(sq) > 0.05                              # the lower lid takes part


def test_held_lids_get_no_blink_extras():
    rg = junor()
    for i in range(40):
        f = rg.solve(vec(blink_l=0.3, blink_r=0.3), t=i/24, cam_pos=CAM)
    assert _w(rg, f, "browDownLeft") < 1e-6


def test_lids_follow_gaze_pitch(r):
    f0 = r.solve(vec(), t=0.0, cam_pos=CAM)
    fd = r.solve(vec(gaze_y=-1.0), t=0.0, cam_pos=CAM)
    fu = r.solve(vec(gaze_y=1.0), t=0.0, cam_pos=CAM)
    assert _w(r, fd, "eyeBlinkLeft") > _w(r, f0, "eyeBlinkLeft") + 0.05
    assert _w(r, fu, "eyeWideLeft") > _w(r, f0, "eyeWideLeft") + 0.03


def test_bilabial_seals_and_lip_timing_is_not_delayed():
    """partial presses seal; the jaw curve of a lip sequence keeps its frame timing (no filter, no look-ahead)."""
    a = junor(); b = RIG.JunoRig(ASSET)
    rng = np.random.default_rng(3)
    lo = np.clip(np.convolve(rng.uniform(0, 1, 120), np.ones(4)/4, "same"), 0, 1)
    ja, jb = [], []
    for i, v in enumerate(lo):
        P = vec(lip_open=v, teeth_gap=v, voice=0.6)
        ja.append(_w(a, a.solve(P, t=i/24, cam_pos=CAM), "jawOpen")); jb.append(_w(b, b.solve(P, t=i/24, cam_pos=CAM), "jawOpen"))
    ja, jb = np.array(ja), np.array(jb)
    assert np.allclose(ja, jb, atol=1e-6)               # same jaw, frame for frame
    f = a.solve(vec(lip_press=0.35, teeth_gap=0.3, lip_open=0.3), t=10.0, cam_pos=CAM)
    assert _w(a, f, "jawOpen") < 0.12 and _w(a, f, "lipsTogether") > 0.6


def test_head_turn_shared_down_the_spine_and_hands_stay(r):
    rg = junor()
    P0 = vec(hand_r_y=0.6, hand_r_z=0.4, hand_l_y=-0.1)
    for i in range(30):
        f0 = rg.solve(P0, t=i/24, cam_pos=CAM)
    P = vec(hand_r_y=0.6, hand_r_z=0.4, hand_l_y=-0.1, head_yaw=0.6)
    for i in range(30):
        f = rg.solve(P, t=2 + i/24, cam_pos=CAM)
    bi = rg.bi
    Rch = f.bones_local[bi["spine_03"], :3, :3].astype(float) @ np.linalg.inv(rg.Mrest_lin[bi["spine_03"]])/rg.k
    yaw = math.atan2(Rch[0, 2], Rch[2, 2])
    assert 0.10*0.6 < yaw < 0.20*0.6, yaw              # the chest takes ~15 %
    Rh = f.bones_local[bi["head"], :3, :3].astype(float) @ np.linalg.inv(rg.Mrest_lin[bi["head"]])/rg.k
    assert abs(math.atan2(Rh[0, 2], Rh[2, 2]) - 0.6) < 0.03     # the head still looks where the page asked
    for s in range(2):
        assert np.linalg.norm(f.grasp_local[s, :3, 3] - f0.grasp_local[s, :3, 3]) < 0.004


def test_breath_moves_the_chest_volume(r):
    bi = r.bi
    a = r.solve(vec(breath=1.0), t=50.0, cam_pos=CAM); b = r.solve(vec(breath=-1.0), t=50.05, cam_pos=CAM)
    sa = np.linalg.norm(a.bones_local[bi["spine_03"], :3, 2]); sb = np.linalg.norm(b.bones_local[bi["spine_03"], :3, 2])
    assert sa != sb


def test_expression_slew_has_no_snap():
    rg = junor()
    vals = []
    for i in range(12):
        f = rg.solve(vec(brow_l=1.0 if i >= 4 else 0.0), t=i/24, cam_pos=CAM)
        vals.append(_w(rg, f, "browOuterUpRight"))
    d = np.diff(vals)
    assert d.max() < 0.75*max(vals), d                  # a step arrives over a few frames


def test_cost(r):
    P = vec(smile=0.3, lip_open=0.4, teeth_gap=0.4, voice=0.8, breath=0.5, head_yaw=0.3, hand_r_y=0.9)
    j = RIG.JunoRig(ASSET)
    for k in range(5):
        r.solve(P, t=k/24); j.solve(P, t=k/24)
    tj = tr = 0.0
    for k in range(40):
        t0 = time.process_time(); j.solve(P, t=1 + k/24); tj += time.process_time() - t0
        t0 = time.process_time(); r.solve(P, t=1 + k/24); tr += time.process_time() - t0
    assert tr < tj*1.35 + 0.02, (tr/40*1e3, tj/40*1e3)


def test_globe_retracts_under_a_closing_lid(r):
    a = r.solve(vec(), t=200.0, cam_pos=CAM); b = r.solve(vec(blink_l=0.6, blink_r=0.6), t=200.04, cam_pos=CAM)
    d = (a.eyes - b.eyes).astype(float)
    mm = np.linalg.norm(d, axis=1)/r.k*1e3
    assert np.all(mm > 0.8) and np.all(mm < 2.5), mm


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
