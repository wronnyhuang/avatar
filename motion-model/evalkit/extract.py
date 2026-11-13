#!/usr/bin/env python3
"""Live-condition user input for one eval clip: native MediaPipe at the live page's rates, fed to BOTH consumers.

  python -I evalkit/extract.py [--ids a,b] [--jobs 6]          (clips from $T/eval/clips/manifest.json)

Per clip (window [v_t0, v_t0 + duration_s] of the user's video; tail beyond the source = the last frame held):
  camera.mp4        the user's camera for the window (<=640 px wide, 30 fps, the user's mic as audio) — sbs + judge context
  mic.wav           the user's voice (mono 24 kHz) ; voice.wav = the avatar's speech for the window (mono 24 kHz)
  perception.jsonl  the CURRENT avatar perception.js fed natively, causally (evalkit/js helper):
                    {t, features, events} per 30 Hz camera frame — the rules / gru-prior input, exactly as the page
  mp_live.npz       the same detections as raw arrays, sample-and-hold per 30 Hz tick, for canon1.fit.Fitter
                    (bs, mat row-major, pose image/world, hands image/world, labels, scores, *_fresh flags)
  audio42_user.npy / audio42_self.npy   canon audio features (spec.audio_features, causal) of mic / voice
Scheduler = the page's credit scheduler (perception.js: face 18 / hand 10 / pose 6 Hz, credit <= 1.5), frames
<= 640 px wide (the page asks the camera for 640x480), raw UNMIRRORED frames. This is controlled replay with native
MediaPipe, NOT a browser end-to-end run (no WASM, no camera, no network).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (E, KIT, AVATAR, MP_MODELS, FPS, atomic_json, read_wav_mono, write_wav, sha_file,  # noqa: E402
                    canon_spec)

RATES = {"face": 18, "hand": 10, "pose": 6}
MAX_W = 640
PERCEPTION = AVATAR / "app/perception.js"
HELPER = KIT / "js/native_perception_replay.cjs"


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate", "-of", "json", "--", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), (float(num) / float(den) if float(den) else 30.0)


def frames(path, t0, dur, w, h):
    vf = f"fps={FPS},scale={w}:{h}:flags=area,tpad=stop_mode=clone:stop_duration={dur + 2:.2f}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-ss", f"{t0:.3f}", "-i", str(path), "-t",
           f"{dur:.3f}", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    k = 0
    while k < int(round(dur * FPS)):
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
        k += 1
    p.stdout.close()
    p.kill()
    p.wait()


MIC_ACTIVE_RMS = 0.08     # active-speech median 10 ms rms of the YouTube suite's mics (browser-like level)
VOICE_ACTIVE_RMS = 0.065  # same for the recorded mock-avatar (TTS) voice


def active_rms(x, sr):
    n = int(sr * 0.01)
    b = x[:len(x) // n * n].reshape(-1, n)
    r = np.sqrt((b ** 2).mean(1))
    pk = np.percentile(r, 99) if len(r) else 0.0
    a = r[r > 0.1 * pk]
    return float(np.median(a)) if len(a) else 0.0


def cut_audio(path, t0, dur, sr_out=24000, target=None):
    """window of a wav, mono, resampled; target = normalize the FILE's active-speech rms to this (Seamless tracks are
    denoised and ~20x quieter than a browser mic / TTS stream; the page's thresholds are absolute)."""
    x, sr = read_wav_mono(path)
    if target:
        g = target / max(active_rms(x, sr), 1e-5)
        x = np.clip(x * g, -1, 1)
    a = int(round(t0 * sr))
    seg = x[max(0, a):max(0, a) + int(round(dur * sr))]
    if a < 0:
        seg = np.concatenate([np.zeros(-a, np.float32), seg])
    seg = np.concatenate([seg, np.zeros(max(0, int(round(dur * sr)) - len(seg)), np.float32)])
    if sr != sr_out:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), sr_out)
        seg = resample_poly(seg.astype(np.float64), sr_out // g, int(sr) // g).astype(np.float32)
    return seg


def audio42(x24k, n):
    spec = canon_spec()
    x16 = spec.resample_to_16k(x24k, 24000)
    return spec.audio_features(x16, n_frames=n)


def extract(clip: dict, force=False) -> dict:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    cid = clip["id"]
    od = E / "clips" / cid
    od.mkdir(parents=True, exist_ok=True)
    rec_p = od / "receipt.json"
    if rec_p.exists() and not force:
        return json.loads(rec_p.read_text())
    t_start = time.monotonic()
    dur = float(clip["duration_s"])
    N = int(round(dur * FPS))
    W0, H0, fps0 = probe(clip["video"])
    w = min(MAX_W, W0 - W0 % 2)
    h = int(round(H0 * w / W0 / 2)) * 2
    # ---- audio: the user's mic and the avatar's speech for the window
    norm = clip["group"] == "seamless"
    mic = (cut_audio(clip["user_wav"], clip.get("user_wav_t0", clip["v_t0"]), dur, target=MIC_ACTIVE_RMS if norm else None)
           if clip.get("user_wav") else np.zeros(int(dur * 24000), np.float32))
    voice = (cut_audio(clip["voice_wav"], clip.get("voice_t0", 0.0), dur, target=VOICE_ACTIVE_RMS if norm else None)
             if clip.get("voice_wav") else np.zeros(int(dur * 24000), np.float32))
    write_wav(od / "mic.wav", mic, 24000)
    write_wav(od / "voice.wav", voice, 24000)
    np.save(od / "audio42_user.npy", audio42(mic, N))
    np.save(od / "audio42_self.npy", audio42(voice, N))
    if norm:   # diagnostic: the dataset's own (quiet) levels, as the model saw them in training
        np.save(od / "audio42_user_raw.npy", audio42(cut_audio(clip["user_wav"], clip.get("user_wav_t0", clip["v_t0"]), dur), N))
        np.save(od / "audio42_self_raw.npy", audio42(cut_audio(clip["voice_wav"], clip.get("voice_t0", 0.0), dur), N))
    # ---- camera window for the sbs / judge context (user's mic as its audio)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-ss", f"{clip['v_t0']:.3f}", "-i", str(clip["video"]),
                    "-i", str(od / "mic.wav"), "-map", "0:v:0", "-map", "1:a:0", "-t", f"{dur:.3f}",
                    "-vf", f"fps={FPS},scale={w}:{h},tpad=stop_mode=clone:stop_duration={dur + 2:.2f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-b:a", "64k", "-shortest", str(od / "camera.mp4")], check=True)
    # ---- MediaPipe at the live rates
    opts = lambda fn: BaseOptions(model_asset_path=str(MP_MODELS / fn))  # noqa: E731
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=opts("face_landmarker.task"), running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    hand = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=opts("gesture_recognizer.task"), running_mode=vision.RunningMode.VIDEO, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=opts("pose_landmarker_lite.task"), running_mode=vision.RunningMode.VIDEO, num_poses=1))
    from PIL import Image
    R = {k: [] for k in ("face", "bs", "mat", "pose", "pose_ok", "pose_world", "hands", "hands_world", "hand_label",
                         "hand_n", "hand_score", "face_fresh", "pose_fresh", "hand_fresh")}
    cur = dict(face=0, bs=np.zeros(52, np.float32), mat=np.zeros(16, np.float32), pose=np.zeros((33, 4), np.float32),
               pose_ok=0, pose_world=np.zeros((33, 4), np.float32), hands=np.zeros((2, 21, 3), np.float32),
               hands_world=np.zeros((2, 21, 3), np.float32), hand_label=np.full(2, -1, np.int8), hand_n=0,
               hand_score=np.zeros(2, np.float32))
    credit = {"face": 1.0, "hand": 1.0, "pose": 1.0}
    prev_gray, motion, motion_t, last_t = None, 0.0, -1.0, 0.0
    partial = od / "perception.partial"
    nfaces = nhands = 0
    with partial.open("w") as fo:
        node = subprocess.Popen(["node", str(HELPER), str(PERCEPTION)], stdin=subprocess.PIPE, stdout=fo, text=True)
        try:
            for i, img in enumerate(frames(clip["video"], clip["v_t0"], dur, w, h)):
                t = i / FPS
                ts = int(round(t * 1000)) + 1
                dt = max(0.0, t - last_t); last_t = t
                frame = {"t": ts}
                im = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
                if t - motion_t >= .066:      # perception.js thumbnail motion energy (v2 convention)
                    tiny = np.asarray(Image.fromarray(img).resize((32, 24), Image.Resampling.BILINEAR), dtype=float)
                    gray = tiny @ np.array([.299, .587, .114])
                    if prev_gray is not None:
                        raw = np.clip((np.abs(gray - prev_gray).mean() - 1.2) / 14, 0, 1)
                        alpha = 1 - math.exp(-2 * math.pi * 3 * max(.001, min(.5, t - motion_t)))
                        motion += alpha * (float(raw) - motion)
                    prev_gray = gray; motion_t = t
                frame["motionFrame"] = motion
                fresh = {"face": 0, "hand": 0, "pose": 0}
                for name, rate in RATES.items():
                    credit[name] = min(1.5, credit[name] + dt * rate)
                    if credit[name] < 1:
                        continue
                    credit[name] -= 1
                    fresh[name] = 1
                    common = {"t": ts, "w": w, "h": h}
                    if name == "face":
                        r = face.detect_for_video(im, ts)
                        common["present"] = bool(r.face_landmarks)
                        if common["present"]:
                            M4 = np.asarray(r.facial_transformation_matrixes[0], np.float32).reshape(4, 4)
                            bs = [q.score for q in r.face_blendshapes[0]]
                            common.update(lm=[v for q in r.face_landmarks[0] for v in (q.x, q.y, q.z)], bs=bs,
                                          names=[q.category_name for q in r.face_blendshapes[0]],
                                          mat=M4.flatten(order="F").tolist())
                            cur.update(face=1, bs=np.asarray(bs, np.float32), mat=M4.reshape(16))
                            nfaces += 1
                        else:
                            cur.update(face=0)
                    elif name == "hand":
                        r = hand.recognize_for_video(im, ts)
                        hs = []
                        H_ = np.zeros((2, 21, 3), np.float32); HW = np.zeros((2, 21, 3), np.float32)
                        lab = np.full(2, -1, np.int8); sc = np.zeros(2, np.float32)
                        for j, lm in enumerate(r.hand_landmarks[:2]):
                            hd = r.handedness[j][0]; g = r.gestures[j][0]
                            hs.append({"lm": [v for q in lm for v in (q.x, q.y, q.z)], "label": hd.category_name,
                                       "lscore": hd.score, "gesture": g.category_name, "gscore": g.score})
                            H_[j] = [[q.x, q.y, q.z] for q in lm]
                            if r.hand_world_landmarks and j < len(r.hand_world_landmarks):
                                HW[j] = [[q.x, q.y, q.z] for q in r.hand_world_landmarks[j]]
                            nm = (hd.category_name or "").lower()
                            lab[j] = 0 if nm.startswith("left") else (1 if nm.startswith("right") else -1)
                            sc[j] = hd.score
                        common["hands"] = hs
                        nhands += len(hs)
                        cur.update(hands=H_, hands_world=HW, hand_label=lab, hand_score=sc, hand_n=min(2, len(hs)))
                    else:
                        r = pose.detect_for_video(im, ts)
                        common["present"] = bool(r.pose_landmarks)
                        if common["present"]:
                            common["lm"] = [v for q in r.pose_landmarks[0] for v in (q.x, q.y, q.z, q.visibility)]
                            wl = r.pose_world_landmarks[0] if r.pose_world_landmarks else None
                            cur.update(pose_ok=1, pose=np.array([[q.x, q.y, q.z, q.visibility] for q in r.pose_landmarks[0]], np.float32),
                                       pose_world=(np.array([[q.x, q.y, q.z, q.visibility] for q in wl], np.float32)
                                                   if wl else np.zeros((33, 4), np.float32)))
                        else:
                            cur.update(pose_ok=0)
                    frame[name] = common
                node.stdin.write(json.dumps(frame, separators=(",", ":")) + "\n")
                for k in ("face", "bs", "mat", "pose", "pose_ok", "pose_world", "hands", "hands_world", "hand_label",
                          "hand_n", "hand_score"):
                    R[k].append(np.copy(cur[k]) if isinstance(cur[k], np.ndarray) else cur[k])
                R["face_fresh"].append(fresh["face"]); R["pose_fresh"].append(fresh["pose"]); R["hand_fresh"].append(fresh["hand"])
            node.stdin.close()
            if node.wait(timeout=60):
                raise RuntimeError("node perception replay failed")
        except BaseException:
            node.kill(); node.wait(timeout=10); raise
        finally:
            for m in (face, hand, pose):
                m.close()
    partial.replace(od / "perception.jsonl")
    n = len(R["face"])
    np.savez_compressed(od / "mp_live.npz",
                        face=np.array(R["face"], np.uint8), bs=np.array(R["bs"], np.float32).reshape(n, 52),
                        mat=np.array(R["mat"], np.float32).reshape(n, 16), pose=np.array(R["pose"], np.float32).reshape(n, 33, 4),
                        pose_ok=np.array(R["pose_ok"], np.uint8), pose_world=np.array(R["pose_world"], np.float32).reshape(n, 33, 4),
                        hands=np.array(R["hands"], np.float32).reshape(n, 2, 21, 3),
                        hands_world=np.array(R["hands_world"], np.float32).reshape(n, 2, 21, 3),
                        hand_label=np.array(R["hand_label"], np.int8).reshape(n, 2), hand_n=np.array(R["hand_n"], np.uint8),
                        hand_score=np.array(R["hand_score"], np.float32).reshape(n, 2),
                        face_fresh=np.array(R["face_fresh"], np.uint8), pose_fresh=np.array(R["pose_fresh"], np.uint8),
                        hand_fresh=np.array(R["hand_fresh"], np.uint8), wh=np.array([w, h]), fps=np.float64(FPS))
    rec = {"id": cid, "frames": n, "expected_frames": N, "face_detections": nfaces, "hand_detections": nhands,
           "face_frac": round(float(np.mean(R["face"])) if n else 0.0, 3), "wh": [w, h], "src_wh": [W0, H0],
           "src_fps": fps0, "rates_hz": RATES, "perception_sha": sha_file(PERCEPTION), "seconds": round(time.monotonic() - t_start, 1),
           "audio_level_norm": bool(norm), "mode": "native MediaPipe at the page's rates + exact JS perception (causal replay); not browser E2E"}
    atomic_json(rec_p, rec)
    return rec


def load_manifest():
    return json.loads((E / "clips/manifest.json").read_text())["clips"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids"); ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    clips = load_manifest()
    if a.ids:
        want = set(a.ids.split(","))
        clips = [c for c in clips if c["id"] in want]
    with cf.ProcessPoolExecutor(a.jobs) as pool:
        futs = {pool.submit(extract, c, a.force): c["id"] for c in clips}
        for f in cf.as_completed(futs):
            try:
                print(json.dumps(f.result()), flush=True)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"id": futs[f], "error": repr(e)}), flush=True)


if __name__ == "__main__":
    main()
