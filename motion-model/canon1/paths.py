"""Data / output locations for the motion-model tools, overridable by environment variables.

Defaults are relative to the motion-model directory, so a fresh checkout works without any setup:

  MM_DATA_DIR   processed data                          default <motion-model>/data
                  canon1/<dataset>/*.npz     CANON v1 sessions (tools/data converters)
                  splits.json                person-disjoint splits (canon1/splits.py)
                  fit/fitter_params.npz      MediaPipe -> CANON fitter parameters (canon1/fit.py)
                  fit/noise_profile.json     fitter noise profile (training augmentation)
                  mapper/                    Seamless -> MediaPipe face mapper (canon1/react_v1)
  MM_WORK_DIR   outputs and intermediate artifacts       default <motion-model>/work
                  runs/                      training runs and exported bundles
                  eval/                      evalkit clips, traces, reports
                  p2/                        packs, audio features, speech-encoder weights, scorecards
  DATASETS_DIR  raw public datasets                     default <motion-model>/datasets
                  seamless-interaction/  learning2listen/  cmu-panoptic/
                  (each also overridable: SEAMLESS_DIR, L2L_DIR, PANOPTIC_DIR)
"""
import os
from pathlib import Path

MM_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MM_DATA_DIR", str(MM_ROOT / "data")))
WORK = Path(os.environ.get("MM_WORK_DIR", str(MM_ROOT / "work")))
DATASETS = Path(os.environ.get("DATASETS_DIR", str(MM_ROOT / "datasets")))
SEAMLESS = Path(os.environ.get("SEAMLESS_DIR", str(DATASETS / "seamless-interaction")))
L2L = Path(os.environ.get("L2L_DIR", str(DATASETS / "learning2listen")))
PANOPTIC = Path(os.environ.get("PANOPTIC_DIR", str(DATASETS / "cmu-panoptic")))
P2 = WORK / "p2"
NOISE_PROFILE = DATA / "fit" / "noise_profile.json"
