"""head-lod — shrink the built junor.asset textures without visible loss (run last, rewrites material paths).

 * lod1 maps of the wardrobe (cloth_* / shoes) were byte copies of the 1024 lod0 maps -> half size (max 512): lod1 is the
   hub's 270p lane.
 * albedo maps (opaque alpha) >= MIN_BYTES -> JPEG q95, 4:4:4 (rmse ~1.3/255 on the skin atlas, invisible);
   normal / packed maps >= MIN_BYTES -> lossless WebP (bit-exact). lod0 and _1k files keep the same extension.
 The renderer opens maps through PIL (format by content); character.py / asset_io.py apply the lod tex_suffix before
 the extension for non-.png paths (a .png path takes the old code path unchanged).
 Env JUNOR_TEX_PACK=0 skips the pass (v6 behaviour).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

MIN_BYTES = 200_000
WARDROBE_LOD1_MAX = 512


def _suffixed(p, suf):
    r, e = os.path.splitext(p)
    return r + suf + e


def finalize(out, mats, suf="_1k", log=print):
    texdir = os.path.join(out, "tex")
    before = sum(os.path.getsize(os.path.join(texdir, f)) for f in os.listdir(texdir))
    done = {}
    for mid, md in mats.items():
        for key in ("albedo", "normal", "packed"):
            rel = md.get(key)
            if not rel or not rel.endswith(".png"):
                continue
            if rel in done:
                md[key] = done[rel]
                continue
            p0 = os.path.join(out, rel)
            p1 = _suffixed(p0, suf)
            if not os.path.exists(p0):
                continue
            ims = {p0: Image.open(p0)}
            ims[p0].load()
            if os.path.exists(p1):
                ims[p1] = Image.open(p1)
                ims[p1].load()
                wardrobe = mid.startswith("cloth_") or mid.startswith("shoes")
                im1 = ims[p1]
                if wardrobe and max(im1.size) > WARDROBE_LOD1_MAX:
                    k = max(im1.size) // WARDROBE_LOD1_MAX
                    ims[p1] = im1.resize((im1.width // k, im1.height // k), Image.LANCZOS)
                    ims[p1].save(p1, compress_level=6)
            if os.path.getsize(p0) < MIN_BYTES:
                done[rel] = rel
                continue
            im0 = ims[p0]
            if key == "albedo":
                if im0.mode == "RGBA" and int(np.asarray(im0)[..., 3].min()) < 255:
                    done[rel] = rel
                    continue
                ext, kw = ".jpg", dict(format="JPEG", quality=95, subsampling=0)
            else:
                ext, kw = ".webp", dict(format="WEBP", lossless=True, method=4)
            for p, im in ims.items():
                q = os.path.splitext(p)[0] + ext
                (im.convert("RGB") if ext == ".jpg" else im).save(q, **kw)
                os.remove(p)
            md[key] = done[rel] = os.path.splitext(rel)[0] + ext
    after = sum(os.path.getsize(os.path.join(texdir, f)) for f in os.listdir(texdir))
    log(f"tex_pack: tex {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    return mats
