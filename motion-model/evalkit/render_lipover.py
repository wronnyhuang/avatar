#!/usr/bin/env python3
"""render_timeline.py (avatar repo, unchanged) with a lip source that REPLACES lipstream's columns where an override
says so (Audio2Face-3D drives the Muse's mouth while the avatar speaks).

  <render venv python> -B evalkit/render_lipover.py <lip_override.npz> -- <render_timeline.py args>   (cwd = renderer/)

lip_override.npz: cols (k,) Muse lip column names, values (N, k) per 30 fps row, w (N,) 0..1. Each lipstream row i
(30 fps; the render's frames pick rows by its own resampler) gets row[cols] += (values[i] - row[cols]) * w[i] BEFORE the
page rig is combined (jaw = max(lip, mouth_open) etc. unchanged) -- exactly evalkit/onscreen.apply_override.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]     # evalkit names must not shadow the renderer's
REND = os.getcwd()
sys.path[:0] = [REND, os.path.join(REND, "engine"), os.path.join(REND, "tools")]

import numpy as np  # noqa: E402


def main():
    if len(sys.argv) < 3 or sys.argv[2] != "--":
        raise SystemExit(__doc__)
    z = np.load(sys.argv[1], allow_pickle=False)
    cols = [str(c) for c in z["cols"]]
    vals = z["values"].astype(np.float32)
    w = z["w"].astype(np.float32)
    import avatar.render_server as RS
    import render_timeline as RT
    from anim.schema import IDX
    ix = np.array([IDX[c] for c in cols], int)

    class LipCore(RS.Core):
        def _take(self, frames):
            for f in frames:
                i = int(f.index)
                if i < len(w) and w[i] > 0:
                    p = np.array(f.params, np.float32, copy=True)
                    p[ix] = p[ix] + (vals[i] - p[ix]) * w[i]
                    f.params = p
            return super()._take(frames)

    RS.Core = LipCore
    sys.argv = [os.path.join(REND, "tools", "render_timeline.py")] + sys.argv[3:]
    RT.main()


if __name__ == "__main__":
    main()
