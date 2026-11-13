"""Learning to Listen (L2L, Ng et al. CVPR 2022) -> CANON v1 (auxiliary head-only source).

What L2L ships (see the release's README.md under $L2L_DIR): 64-frame windows (30 fps per the release) of DECA/FLAME
parameters for the two people of a talk-show / podcast two-shot (p0 = left of frame, p1 = right), in both directions
(p0 speaking + p1 listening, and the reverse), plus a 128-bin log-mel of the SPEAKER only. No raw audio, no video,
no body, no gaze.

Conversion
  sessions   all windows of one identity (train/ + test/, both directions) are merged per source video by frame index
             (overlapping windows carry identical values - verified), then cut into contiguous runs; a run is also cut
             where either person's DECA head rotation jumps by more than CUT_RAD in one frame (camera cut / tracking
             swap). Runs shorter than MIN_FRAMES are dropped. One session = one run; P = 2.
  persons    person 0 = the channel's named host (the L2L "listener identity"), person 1 = the other person (guest).
             Host side: p0 for conan/kimmel/stephen/trevor, p1 for fallon (authors' official listener side) and devi2
             (DECA detail-code consistency across videos; see HOST_SIDE). Kimmel guest-host episodes get their own id.
  head       FLAME global rotation (DECA pose[0:3], camera frame; FLAME axes = SMPL axes: x = person's left, y = up,
             z = forward) -> rotation relative to the run's chordal median (spec.relative_root) -> light Gaussian
             smoothing (SMOOTH_SIGMA frames) -> split between neck and head with the per-axis shares measured on Seamless
             SMPL-H (NECK_SHARE) and composed onto the Seamless population median neck / head pose (NECK0 / HEAD0):
                neck = log(R(NECK0) R(NECK_SHARE * d)),  head = log(R(HEAD0) R((1 - NECK_SHARE) * d)).
             No torso is observed, so trunk motion relative to the camera is folded into neck+head.
             Sign check (empirical): DECA jaw x-rotation is > 0 in 85-96 % of frames and larger while speaking than
             while listening for every side of every identity -> +x = chin down (SMPL: + rot about x = nod down).
  masks      head = 1 on every frame of a run (both people); every other group = 0 (values = Seamless population
             median = the neutral fill). FLAME expression does not map honestly to ARKit blendshapes -> face_upper 0;
             mouth needs every mouth*/jaw*/cheekPuff column -> 0 (jawOpen left neutral; raw FLAME kept in x_flame).
  audio      L2L ships only a 128-bin log-mel of the speaker (not the 40-bin causal features, not re-computable
             without the waveform) -> audio = zeros, loud_db = -100, vad = 0 for both people; meta audio = "missing".
             The dataset's speaker/listener role per frame is kept in x_role (P,T) int8 (1 = designated speaker of the
             window covering the frame, 0 = designated listener, -1 = covered by windows of both directions).
  extras     x_flame (P,T,56) float16 = raw DECA [expression 50 | global rot 3 | jaw rot 3] per frame;
             x_role (P,T) int8; x_energy (P,T,3) float16 (same definition as Seamless).
  fps        the release states 30 fps (64-frame windows = 2.13 s, 4 mel frames per video frame). Source videos are
             YouTube uploads (NTSC 29.97 or 30); the 0.1 % difference is below what can be resolved -> no resampling.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import spec as S

from . import paths as _paths  # noqa: E402

ROOT = _paths.L2L                                     # $L2L_DIR (or $DATASETS_DIR/learning2listen)
CONVERTER_VERSION = "l2l-canon1-v1"
IDENTITIES = ["conan", "devi2", "fallon", "kimmel", "stephen", "trevor"]
# side (0 = left of frame, 1 = right) of the named host. conan/kimmel/stephen/trevor/fallon = the authors' official
# listener side; devi2 (unused by the authors) from DECA detail-code consistency across videos (median distance of the
# per-video mean detail code to its median: p1 0.089 vs p0 0.142). Same test confirms p0 for conan .154/.235,
# kimmel .181/.267, stephen .145/.231, trevor .138/.263; fallon is inconclusive (.195/.201) -> authors' p1.
HOST_SIDE = {"conan": 0, "kimmel": 0, "stephen": 0, "trevor": 0, "fallon": 1, "devi2": 1}
LABEL = {"devi2": "podcast-interview"}
CUT_RAD = 0.30            # per-frame head rotation jump (either person) treated as a cut
MIN_FRAMES = 64           # one L2L window
SMOOTH_SIGMA = 1.0        # frames, Gaussian on the relative head rotation vector (DECA is per-frame and jittery)
# Seamless SMPL-H (135 shards, 410k frames): regression of neck / head deviation on the composite (neck*head)
# deviation per axis: neck slopes x .588 y .568 z .528, head .414 .434 .445 -> neck share normalised:
NECK_SHARE = np.array([0.587, 0.567, 0.543])
# Seamless population median (valid frames) of the neck / head local rotations (axis-angle, rad)
NECK0 = np.array([-0.225, -0.003, 0.003])
HEAD0 = np.array([0.064, 0.0, -0.008])
LICENCE = ("no licence stated by the authors (Learning to Listen, UC Berkeley); features of copyrighted YouTube videos; "
           "DECA/FLAME non-commercial research licence -> research use only, do not redistribute")


# ----------------------------------------------------------------------------------------------- neutral fill
def seamless_neutral(cache, seamless_dir=None, every=12):
    """(189,) Seamless population median of every column over its valid frames (root = 0). Cached as .npy."""
    cache = Path(cache)
    if cache.exists():
        return np.load(cache, allow_pickle=False).astype(np.float64)
    G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
    Ms, Ks = [], []
    if seamless_dir is None:
        seamless_dir = _paths.DATA / "canon1" / "seamless"
    for f in sorted(Path(seamless_dir).glob("V*.npz"))[::every]:
        d = np.load(f, allow_pickle=False)
        Ms.append(d["motion"][:, ::5].reshape(-1, S.D_MOTION).astype(np.float32))
        Ks.append(d["mask"][:, ::5].reshape(-1, S.D_MASK))
    M, K = np.concatenate(Ms), np.concatenate(Ks)
    med = np.zeros(S.D_MOTION)
    for g, cols in S.GROUPS.items():
        v = K[:, G[g]] > 0
        if v.any():
            med[cols] = np.median(M[v][:, cols], 0)
    med[0:6] = 0.0
    tmp = cache.with_name(cache.name + ".tmp.npy")
    np.save(tmp, med)
    os.replace(tmp, cache)
    return med


# ----------------------------------------------------------------------------------------------- loading
def _load(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_identity(ident, feat_dim=56):
    """-> dict video -> dict(f0 (R,56), f1 (R,56), have (R,) bool, role (R,2) counts of windows naming person s as
    speaker, frame0 int). All windows of train/ and test/ (symlinked copies read once), both directions."""
    seen = set()
    chunks = []
    for split in ("train", "test"):
        d = ROOT / ident / split
        for s in (0, 1):
            fp = d / f"p{s}_speak_files_clean_deca.npy"
            key = os.path.realpath(fp)
            if not fp.exists() or key in seen:
                continue
            seen.add(key)
            files = _load(fp)
            sp = _load(d / f"p{s}_speak_faces_clean_deca.npy")
            li = _load(d / f"p{1 - s}_list_faces_clean_deca.npy")
            assert len(files) == len(sp) == len(li), (ident, split, s)
            vid = np.asarray(files[:, 0, 0])
            st = np.asarray(files[:, :, 2]).astype(np.int64)
            assert (np.diff(st, axis=1) == 1).all()
            chunks.append((s, vid, st[:, 0], np.asarray(sp[:, :, :feat_dim], np.float32),
                           np.asarray(li[:, :, :feat_dim], np.float32)))
    byvid = defaultdict(list)
    for ci, (s, vid, st0, _, _) in enumerate(chunks):
        for i, v in enumerate(vid):
            byvid[str(v)].append((ci, i))
    out = {}
    for v, lst in byvid.items():
        st = np.array([chunks[ci][2][i] for ci, i in lst])
        f0 = int(st.min()); R = int(st.max()) + 64 - f0
        F = np.zeros((2, R, feat_dim), np.float32)
        have = np.zeros(R, bool)
        role = np.zeros((R, 2), np.int16)
        for ci, i in lst:
            s, _, st0, sp, li = chunks[ci]
            a = int(st0[i]) - f0
            F[s, a:a + 64] = sp[i]
            F[1 - s, a:a + 64] = li[i]
            have[a:a + 64] = True
            role[a:a + 64, s] += 1
        out[v] = dict(F=F, have=have, role=role, frame0=f0, n_windows=len(lst))
    return out


def host_side_for_video(ident, vid):
    return HOST_SIDE[ident]


def person_ids(ident, vid):
    """(host id, guest id). Kimmel guest-host episodes ('GuestHost<Name>Interviews...') get the guest host's name."""
    stem = vid.split("/")[-1]
    host = f"l2l:{ident}"
    m = re.search(r"GuestHost([A-Za-z]+?)Interviews", stem)
    if ident == "kimmel" and m:
        host = f"l2l:kimmel-guesthost:{m.group(1)}"
    slug = re.sub(r"[^A-Za-z0-9]+", "", stem)[:48]
    h = hashlib.sha1(vid.encode()).hexdigest()[:6]
    return host, f"l2l:{ident}-guest:{slug}-{h}"


# ----------------------------------------------------------------------------------------------- segments
def _rot_jump(pose_aa):
    R = S.aa_to_mat(pose_aa)
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    c = np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.arccos(c)


def segments(rec, cut_rad=CUT_RAD, min_frames=MIN_FRAMES):
    """contiguous runs of covered frames, cut at head-rotation jumps. -> list of (a, b) (b exclusive)."""
    have = rec["have"]
    R = len(have)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], have.astype(np.int8), [0]])))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 2:
            continue
        j = np.maximum(_rot_jump(rec["F"][0, a:b, 50:53]), _rot_jump(rec["F"][1, a:b, 50:53]))
        cuts = np.flatnonzero(j > cut_rad) + 1
        bounds = [a] + [a + int(c) for c in cuts] + [b]
        for x, y in zip(bounds[:-1], bounds[1:]):
            if y - x >= min_frames:
                out.append((int(x), int(y)))
    assert all(0 <= a < b <= R for a, b in out)
    return out


# ----------------------------------------------------------------------------------------------- conversion
def head_neck(global_aa):
    """(T,3) FLAME global head rotation (camera frame) -> (T,3) neck, (T,3) head local axis-angles (see module doc)."""
    d = S.relative_root(np.asarray(global_aa, np.float64))
    if SMOOTH_SIGMA > 0 and len(d) > 4:
        d = gaussian_filter1d(d, SMOOTH_SIGMA, axis=0, mode="nearest")
    neck = S.mat_to_aa(S.aa_to_mat(NECK0)[None] @ S.aa_to_mat(d * NECK_SHARE))
    head = S.mat_to_aa(S.aa_to_mat(HEAD0)[None] @ S.aa_to_mat(d * (1.0 - NECK_SHARE)))
    return neck, head, d


def energy_tracks(motion, mask):
    """(T,189),(T,9) -> (T,3): head angular speed, arm joint speed, blendshape L1 speed; as canon1.seamless."""
    G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
    m = motion.astype(np.float64)
    d = np.diff(m, axis=0, prepend=m[:1]) * S.FPS
    head = np.linalg.norm(d[:, S.GROUPS["head"]], axis=1) * (mask[:, G["head"]] > 0)
    arms = np.linalg.norm(d[:, S.GROUPS["arm_l"] + S.GROUPS["arm_r"]], axis=1) * \
        ((mask[:, G["arm_l"]] > 0) & (mask[:, G["arm_r"]] > 0))
    face = np.abs(d[:, S.SL_FACE]).sum(1) * (mask[:, G["face_upper"]] > 0)
    e = np.stack([head, arms, face], 1)
    e = np.minimum(e, np.percentile(e, 99.5, axis=0) + 1e-6)
    return gaussian_filter1d(e, 15, axis=0, mode="nearest")


def session_stats(motion, mask, audio, speaking=None):
    """per-person summary for index.jsonl (keys as canon1.seamless.session_stats; speaking_frac from `speaking`
    (P,T) when the audio vad column is not measured)."""
    out = []
    G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
    for p in range(len(motion)):
        e = energy_tracks(motion[p], mask[p])
        hv = mask[p][:, G["head"]] > 0
        av = (mask[p][:, G["arm_l"]] > 0) & (mask[p][:, G["arm_r"]] > 0)
        fv = mask[p][:, G["face_upper"]] > 0
        bs = motion[p][:, S.SL_FACE]
        spk = audio[p][:, 41] if speaking is None else speaking[p]
        out.append({
            "head_speed_med": round(float(np.median(e[hv, 0])) if hv.any() else 0.0, 4),
            "head_speed_p90": round(float(np.percentile(e[hv, 0], 90)) if hv.any() else 0.0, 4),
            "arm_speed_med": round(float(np.median(e[av, 1])) if av.any() else 0.0, 4),
            "bs_speed_med": round(float(np.median(e[fv, 2])) if fv.any() else 0.0, 4),
            "bs_var_mean": round(float(bs[fv].var(0).mean()) if fv.sum() > 10 else 0.0, 6),
            "still_frac": round(float((e[hv, 0] < 0.05).mean()) if hv.any() else 1.0, 4),
            "speaking_frac": round(float((np.asarray(spk) > 0.5).mean()), 4),
            "smile_mean": round(float(bs[fv][:, [S.BS["mouthSmileLeft"], S.BS["mouthSmileRight"]]].mean())
                                if fv.any() else 0.0, 4),
        })
    return out


def convert_segment(ident, vid, rec, a, b, neutral, out_dir, split="train"):
    """one run [a,b) of one video -> session npz; returns the index row."""
    T = b - a
    host_side = host_side_for_video(ident, vid)
    order = [host_side, 1 - host_side]                       # person 0 = host, person 1 = guest
    pids = list(person_ids(ident, vid))
    G = {g: i for i, g in enumerate(S.GROUP_NAMES)}
    motion = np.tile(neutral.astype(np.float64), (2, T, 1))
    mask = np.zeros((2, T, S.D_MASK), np.uint8)
    audio = np.zeros((2, T, S.D_AUDIO), np.float32)
    audio[..., 40] = -100.0
    flame = np.zeros((2, T, 56), np.float32)
    role = np.zeros((2, T), np.int8)
    head_dev = []
    for p, side in enumerate(order):
        Fp = rec["F"][side, a:b]
        neck, head, d = head_neck(Fp[:, 50:53])
        motion[p][:, S.jcols("neck")] = neck
        motion[p][:, S.jcols("head")] = head
        mask[p][:, G["head"]] = 1
        flame[p] = Fp[:, :56]
        r = rec["role"][a:b]
        mine, other = r[:, side] > 0, r[:, 1 - side] > 0
        role[p] = np.where(mine & other, -1, np.where(mine, 1, 0)).astype(np.int8)
        head_dev.append(d)
    if not np.isfinite(motion).all():
        raise ValueError("non-finite motion")
    motion = motion.astype(np.float32)
    energy = np.stack([energy_tracks(motion[p], mask[p]) for p in range(2)]).astype(np.float16)
    spk = (role == 1).astype(np.float32)
    stats = session_stats(motion, mask, audio, speaking=spk)
    for p in range(2):
        stats[p]["role_speaker_frac"] = round(float((role[p] == 1).mean()), 4)
        stats[p]["role_listener_frac"] = round(float((role[p] == 0).mean()), 4)
        stats[p]["head_dev_std"] = [round(float(x), 4) for x in head_dev[p].std(0)]
    f0 = rec["frame0"] + a
    stem = vid.split("/")[-1]
    session = f"{ident}_{re.sub(r'[^A-Za-z0-9]+', '', stem)[:40]}_{hashlib.sha1(vid.encode()).hexdigest()[:6]}_{f0:06d}"
    quality = [1.0, 1.0]                                    # fraction of frames whose own measured channel (head) is valid
    label = LABEL.get(ident, "talkshow-interview")
    meta = dict(version=S.VERSION, dataset="l2l", session=session, persons=pids, split=split,
                source=["flame-head", "flame-head"], quality=quality, speaking_track="none",
                audio="missing", audio_note="L2L ships only a 128-bin log-mel of the speaker; not the causal 40-bin features -> "
                "zeros, loud_db=-100, vad=0 for both; per-frame speaker/listener role from the window labels in x_role",
                label=label, identity=ident, video=vid, video_frame0=int(f0), host_side=int(host_side),
                person_roles=["host", "guest"], converter=CONVERTER_VERSION,
                head_method=dict(source="DECA FLAME global rotation (camera frame), relative to the run's chordal median",
                                 smooth_sigma_frames=SMOOTH_SIGMA, neck_share=NECK_SHARE.tolist(),
                                 neck0=NECK0.tolist(), head0=HEAD0.tolist(), cut_rad=CUT_RAD,
                                 note="no torso observed: trunk motion relative to the camera is folded into neck+head"),
                neutral_fill="Seamless population median of each column (valid frames); root = 0",
                fps_note="release states 30 fps; not resampled",
                valid_frac=[{g: round(float(mask[p][:, i].mean()), 4) for i, g in enumerate(S.GROUP_NAMES)}
                            for p in range(2)],
                stats=stats,
                x_flame="(P,T,56) float16 raw DECA: expression[0:50], global head rot[50:53], jaw rot[53:56] (axis-angle)",
                x_role="(P,T) int8 1 = dataset-designated speaker of the covering window, 0 = listener, -1 = both",
                x_energy="(P,T,3) head ang speed rad/s, arm joint speed rad/s, blendshape L1 speed /s; smoothed 0.5 s",
                licence=LICENCE)
    out_path = Path(out_dir) / f"{session}.npz"
    tmp = out_path.with_name("." + out_path.stem + ".tmp.npz")
    words = [[], []]
    S.save_session(tmp, motion, mask, audio, meta, words=words,
                   extra={"flame": flame.astype(np.float16), "role": role, "energy": energy})
    os.replace(tmp, out_path)
    return dict(path=str(out_path), dataset="l2l", session=session, persons=pids, split=split, frames=T,
                person_hours=round(2 * T / S.FPS / 3600, 5), source=meta["source"], quality=quality, label=label,
                identity=ident, video=vid, video_frame0=int(f0), audio="missing", stats=stats,
                valid_frac=meta["valid_frac"], converter=CONVERTER_VERSION)


def convert_identity(ident, out_dir, neutral, split="train", log=print):
    recs = load_identity(ident)
    rows, n_seg, n_frames_cov, n_frames_kept = [], 0, 0, 0
    for vid in sorted(recs):
        rec = recs[vid]
        n_frames_cov += int(rec["have"].sum())
        for a, b in segments(rec):
            rows.append(convert_segment(ident, vid, rec, a, b, neutral, out_dir, split))
            n_seg += 1
            n_frames_kept += b - a
    info = dict(identity=ident, videos=len(recs), sessions=n_seg, frames_covered=n_frames_cov,
                frames_kept=n_frames_kept, hours_covered=round(n_frames_cov / S.FPS / 3600, 3),
                hours_kept=round(n_frames_kept / S.FPS / 3600, 3))
    log(json.dumps(info))
    return rows, info
