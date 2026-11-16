#!/usr/bin/env python3
"""HAIR — (re)groom a BUILT asset: scalp + fuzz (+ brows / lashes when their builders exist), written into a
hard-linked copy of the asset.  Re-run on every new head build (target < 5 min on the hub CPU).

  python renderer/humans/junor/groom/regroom.py SRC_ASSET OUT_ASSET [--n 120000] [--n1 26000] [--seed 4557]
        [--keep brows_l,brows_r,lashes_upper_l,...]   # copy these groups as they are from SRC (same head only)
        [--no-brows] [--no-lashes]

Writes OUT_ASSET/hair/lod{0,1}_<group>.bin, the json's hair_groups / lods[].hair / hair_* materials, and the
`hair_render` block (the renderer's hair config for this asset: hair.py uses it instead of the character json's
mesh.hair when present — absent on every other asset, so juno and the rest render exactly as before).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import head as headmod  # noqa: E402
import scalp  # noqa: E402

HAIR_RENDER = {
    "model": "v2",
    "_doc": " HAIR: junor's hair render config (renderer/mesh/hair.py reads asset meta hair_render first)",
    "width_k": {"scalp": 1.6, "fuzz": 1.2, "brows": 1.5, "lashes": 1.6},
    "len_k": {"lashes": 1.0},
    "fuzz_frac": 1.0,
    "min_px": 1.0,
    "cap": {"scalp_alpha": 0.35, "scalp_sigma_mm": 8.0, "lashes_alpha": 0.2, "lashes_sigma_mm": 0.5, "brows_alpha": 0.2, "brows_sigma_mm": 0.9},   # hair-bl r3 + lead scalp
    "follow": {"bone": "spine_03"},
    "shadow_keep": 1.5,      # softer hair->skin key shadow (no dark band under the hairline; SHADE 12:17)
    "layer": {"on": True, "tent_px": 0.75, "col_minpx": 1.0, "width_k": {"brows": 1.6, "lashes": 2.4}, "taper_u": 0.8, "cov_k": {"lashes_lower_l": 0.45, "lashes_lower_r": 0.45}},   # HAIR2: analytic-coverage hair layer (true-width strands, soft silhouette; hair.py LAYER_DEFAULTS)
    "r_var": 0.25, "shift_jitter_deg": [0.8, 2.5, 1.0], "beta": 0.28, "gain": [3.5, 0.5, 1.5, 3.0], "r_tint": 0.6,   # HAIR2 r9: less plastic (judge r8)
    "lock_var": [0.3, 0.25, 0.45],   # HAIR2: one coherent band per lock, not streaks
    "amb_spec": 1.6, "fill_r": 0.4, "amb": 1.3,
    "color_mul": {"lashes": [0.35, 0.35, 0.35]},   # HAIR2 r8: short fine lashes stay dark (not a pale fringe)   # HAIR2: the shadow side keeps a soft sheen (form), not a black helmet
    "lod_frac": [[0.6, 1.0], [2.0, 0.55]],   # hair-shade r7: strand fraction by head px/mm on screen (close-up fill budget)      # hair-shade r6: free lengths hang on nods/tilts + inertial lag (collar clearance)
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=72000)
    ap.add_argument("--n1", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=4557)
    ap.add_argument("--keep", default="")
    ap.add_argument("--no-brows", action="store_true")
    ap.add_argument("--no-lashes", action="store_true")
    ap.add_argument("--style", default="lob", choices=sorted(scalp.STYLES))
    ap.add_argument("--render-cfg", help="json file merged over the default hair_render block")
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    H = headmod.Head(a.src)
    print(f"[regroom] head: O {H.O.round(4)} R {H.R:.4f} u {H.u:.4f} d {H.eye_y - H.chin[1]:.4f} "
          f"ears {[round(e['x'], 4) for e in H.ear.values()]} ({time.time() - t0:.1f}s)", flush=True)
    scalp.STYLE.clear(); scalp.STYLE.update(scalp.STYLES[a.style])
    raw = scalp.build(H, rng, n=a.n, n1=a.n1, log=lambda m: print(m, flush=True))
    if not a.no_brows:
        try:
            import brows
            raw.update(brows.build(H, rng, log=lambda m: print(m, flush=True)))
        except ImportError:
            pass
    if not a.no_lashes:
        try:
            import lashes
            raw.update(lashes.build(H, rng, log=lambda m: print(m, flush=True)))
        except ImportError:
            pass
    keep = [g for g in a.keep.split(",") if g]
    if keep:
        src_meta = json.load(open(headmod.asset_json(a.src)))
        from humans import asset_io
        for g in keep:
            gm = src_meta["hair_groups"][g]
            ent = {"meta": {k: v for k, v in gm.items() if k not in ("lod", "lod_strands", "strands", "points",
                                                                      "root_dist_max_mm")}}
            for lod in ("lod0", "lod1"):
                h = asset_io.load_hair(os.path.join(a.src, gm["lod"][lod]))
                ent[lod] = {k: np.array(h[k]) for k in ("pos", "width", "seed", "clump")}
            raw[g] = ent
    hr = dict(HAIR_RENDER)
    hr["style"] = a.style
    # strand lashes replace the head build's lash cards (character.py skips hair_render.hide_submeshes)
    if any(g.startswith("lashes") for g in raw):
        hr["hide_submeshes"] = [s["name"] for s in H.A.submeshes if s.get("material") == "lashes"]
    if a.render_cfg:
        hr.update(json.load(open(a.render_cfg)))
    groups = headmod.write_groups(a.src, a.out, raw, hair_render=hr)
    print(json.dumps({"out": a.out, "groups": {g: [m["lod_strands"]["lod0"], m["lod_strands"]["lod1"], m["points"],
                                                   m["root_dist_max_mm"]] for g, m in groups.items()},
                      "t": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
