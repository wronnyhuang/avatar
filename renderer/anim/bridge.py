"""The animals' seven old mouth columns, derived from the lip columns so that no loudness reaches them.
A rig that reads lip_* directly ignores these."""
from __future__ import annotations

import numpy as np

from .schema import IDX


def legacy_from_lips(v: np.ndarray) -> np.ndarray:
    """v: (..., NPARAMS), lip columns filled.  Fills jaw, round, spread, press, labdent, tongue in place and returns v."""
    c = lambda n: v[..., IDX[n]]
    width = c("lip_width")
    pk = np.maximum(-width, 0.0)
    v[..., IDX["jaw"]] = np.maximum(c("teeth_gap"), 0.9 * c("lip_open"))
    v[..., IDX["round"]] = np.clip(0.55 * pk + 0.65 * c("lip_protrude"), 0.0, 1.0)
    v[..., IDX["spread"]] = np.maximum(width, 0.0)
    v[..., IDX["press"]] = c("lip_press")
    v[..., IDX["labdent"]] = c("lip_tuck")
    v[..., IDX["tongue"]] = np.maximum(c("tongue_tip"), c("tongue_dental"))
    return v
