#!/usr/bin/env python3
"""RIG: a canon1 motion clip -> an mp4 of Juno (offline), through the live renderer (MeshEngine + JunoRig canon
drive).  A thin front end on tools/render_timeline.py (same engine, same server face life / lip sync with --audio).

  AV_PY -I renderer/tools/render_canon_juno.py --canon clip.npy --character junor|juno --out clip.mp4
      [--w 640 --h 360] [--backend cpu|gpu] [--threads 4] [--audio a.wav] [--weights torso=1,head=1,arm_l=1,...]
      [--head-mode abs|add] [--torso-mode abs|add] [--base rig.jsonl] [--cam home|wide] [--fps 30] [--duration S]
      [--canon-person 0|1 --canon-start FRAME   (Seamless .npz: key motion (2,T,189))] [--lod lod0|lod1] [--life on|off]
      [--rebase auto|off]
--rebase auto (default, .npy clips only): a STANDING mocap clip whose median chest lean differs from the (seated
Seamless) canon neutral by > 8 deg is turned once at the root so she does not lean back ~18 deg (humans/canon_drive.py
rebase_lean; body-internal relations kept).  Seamless clips are within the threshold and render unchanged.

Defaults for offline real-vs-generated clips: --character junor, --cam wide, --head-mode abs, --torso-mode abs,
--fps 30, weights all 1, --life face = the server's idle life on the FACE columns only (blinks, gaze, brows); the canon
owns the body, root and camera (--life on = everything incl. its root / camera drift; off = nothing).
--base rig.jsonl = a dial timeline (render_timeline.py format) under the canon (face, props, ...); its lines may carry
"canon" too.  GPU: `flock gpu.lock ...` and only when the hub demo is idle.
CPU: Mesa llvmpipe; set LP_NUM_THREADS (= --threads) and run several clips in parallel rather than one wide.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canon", required=True)
    ap.add_argument("--character", default="junor", choices=["junor", "juno"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=360)
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--audio")
    ap.add_argument("--weights")
    ap.add_argument("--head-mode", default="abs", choices=["abs", "add"])
    ap.add_argument("--torso-mode", default="abs", choices=["abs", "add"])
    ap.add_argument("--base")
    ap.add_argument("--cam", default="wide", choices=["home", "wide"])
    ap.add_argument("--fps", type=int, default=30, choices=[20, 24, 30])
    ap.add_argument("--duration", type=float)
    ap.add_argument("--canon-person", type=int, default=0)
    ap.add_argument("--canon-start", type=int, default=0)
    ap.add_argument("--lod")
    ap.add_argument("--life", default="face", choices=["face", "on", "off"])
    ap.add_argument("--preset", default="auto")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--rebase", default="auto", choices=["auto", "off"])
    a = ap.parse_args(argv)
    if a.rebase == "auto" and a.canon.endswith(".npy"):
        import numpy as np
        sys.path.insert(0, os.path.dirname(HERE))
        from humans.canon_drive import rebase_lean
        m, deg = rebase_lean(np.load(a.canon))
        if deg:
            import tempfile
            fd, tmp = tempfile.mkstemp(suffix=".npy", prefix="rebased_"); os.close(fd)
            np.save(tmp, m.astype(np.float32)); a.canon = tmp
            print(f"[render_canon_juno] rebase: standing clip, root turned {deg:+.1f} deg about x (median chest lean -> canon neutral)")
    os.environ.setdefault("LP_NUM_THREADS", str(a.threads))
    tl = ["--character", a.character, "--out", a.out, "--w", str(a.w), "--h", str(a.h), "--backend", a.backend,
          "--threads", str(a.threads), "--canon", a.canon, "--head-mode", a.head_mode, "--torso-mode", a.torso_mode,
          "--cam", a.cam, "--fps", str(a.fps), "--canon-person", str(a.canon_person), "--canon-start", str(a.canon_start),
          "--life", a.life, "--preset", a.preset, "--crf", str(a.crf)]
    for k, v in (("--audio", a.audio), ("--weights", a.weights), ("--rig", a.base), ("--lod", a.lod)):
        if v:
            tl += [k, v]
    if a.duration:
        tl += ["--duration", str(a.duration)]
    import render_timeline
    sys.argv = [os.path.join(HERE, "render_timeline.py")] + tl
    render_timeline.main()


if __name__ == "__main__":
    main()
