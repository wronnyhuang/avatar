# junor.asset: sources and licences

Every upstream source of the `junor` ("Juno · realistic") character, its licence, what was changed, and whether the
files are in this repository or must be fetched. Full notice texts are in [`THIRD_PARTY.md`](../../../../THIRD_PARTY.md).
The licence texts below were checked against the upstream repositories and pages (links in the table).

| Part | Source | Licence | What was changed | In this repo? |
|---|---|---|---|---|
| Head geometry, identity (PCA modes), 53 expression shapes + 4 extras, teeth, gums/tongue, mouth socket, lash cards, UV layout | ICT-FaceKit, USC Institute for Creative Technologies Vision & Graphics Lab, https://github.com/ICT-VGL/ICT-FaceKit (`FaceXModel/*.obj`) | MIT, "Copyright (c) 2020 USC Institute for Creative Technologies" | Identity is a blend of PCA modes fitted to a designed face (`identity.npy`, `neutral_concept.npy`); expressions renamed to ARKit, combined shapes (browInnerUp, cheekPuff) and correctives derived; tongue front lowered 5 mm; head grafted onto the body at the ICT neck loop | Derived geometry is inside the built `junor.asset`. The original OBJ files are **fetched** (`get_sources.sh`). |
| Body, skeleton, skin weights | MPFB2 / MakeHuman hm08 base mesh, targets and `game_engine` rig, MPFB2 v2.0.18 data, https://github.com/makehumancommunity/mpfb2 (`src/mpfb/data`) | CC0 1.0 Universal for the assets (`LICENSE.ASSETS.md`); the MPFB2 *source code* is GPLv3, but only its data files are used here, no MPFB2 code | Adult female macro targets (`junor_design.py`), neck warped onto the ICT neck loop, head faces cut, weights re-assigned | Derived body mesh is inside `junor.asset`. The MPFB2 data directory is **fetched** (see below). |
| Eyeballs, cornea, occlusion, tearline, caruncle geometry and eye textures | Original code in this repo (`renderer/humans/juno/build/eyes_geo.py`, `eyes_tex.py`) | Same licence as this repository (MIT) | Fitted to the ICT lids, real iris size (11.8 mm) | Yes |
| Wardrobe geometry and cloth textures | Original code in this repo (`wardrobe_r*.py`, refit for the adult body) | Same licence as this repository | none | Yes |
| Skin pore / cavity / roughness / albedo **detail** (head) | "Infinite, 3D Head Scan" by Lee Perry-Smith (Infinite-Realities, www.ir-ltd.net), based on a work at www.triplegangers.com. Source used: https://casual-effects.com/g3d/data10/research/model/lpshead/lpshead.zip (original release: http://www.ir-ltd.net/infinite-3d-head-scan-released) | CC BY 3.0 Unported, https://creativecommons.org/licenses/by/3.0/ . Attribution is required (see THIRD_PARTY.md). The scan's own licence note adds: "Permissions beyond the scope of this license may be available at http://www.ir-ltd.net/" | Modified: registered to the ICT-FaceKit topology, band-passed (only high-frequency pore and fine-wrinkle detail kept), beard/brow/stretched areas replaced with synthesized pore detail, re-baked into ICT UV space. The base colour is not from the scan. | The derived detail maps are inside `junor.asset/tex`. The scan itself is **fetched** and is not redistributed here. |
| Face base colour (albedo) | Projected from generated photographs of an invented person (image model `google/gemini-3-pro-image` via OpenRouter; prompts in `concept_fit/p_*.txt`, driver `concept_fit/gen.py`), baked with `concept_fit/proj_bake.py`, `finalize.py`, `brow_bake.py` | Generated content; no third-party licence. See "Generated images" in THIRD_PARTY.md | Projected onto the ICT UV atlas, low-frequency colour filled in where unseen | The baked albedo (`skin_head_albedo*.jpg`) is inside `junor.asset/tex`. The source photographs are **not** shipped. |
| Skin base colour elsewhere, freckles, lips, body skin, wrinkle maps | Procedural (`tex_r.py`) | Same licence as this repository | none | Yes (code) |
| Face landmarks used by the concept fit | Google MediaPipe `face_landmarker.task` | Apache-2.0 | Used as a measuring tool only | Not shipped |
| Concept-fit reference (head shape) | TRELLIS.2, Microsoft, https://github.com/microsoft/TRELLIS.2 | MIT | Used only as a shape reference during fitting; nothing from it is shipped | Not shipped |

No MetaHuman assets and no assets from games or films are used anywhere. Rejected sources: Digital Emily 2
(non-commercial licence), FFHQ-UV (CC BY-NC-SA).

## Rebuilding

The built asset is committed at `renderer/humans/junor/junor.asset`, so a rebuild is optional. Steps that still apply:

1. Fetch the ICT-FaceKit sources and convert them:
   ```
   renderer/humans/junor/build/get_sources.sh <work-dir>      # clones ICT-FaceKit, writes <work-dir>/ict.npz via ict_io.py
   ```
2. Fetch the MPFB2 data directory yourself (https://github.com/makehumancommunity/mpfb2, tag v2.0.18, `src/mpfb/data`).
3. Build, regroom the hair, then cull covered cloth triangles:
   ```
   python renderer/humans/junor/build/build_junor.py <mpfb_data_dir> <work-dir>/ict.npz <out_asset_dir> [--identity identity.npy] [--version v1] [--tex 2048]
   python renderer/humans/junor/groom/regroom.py <out_asset_dir> <groomed_asset_dir>
   python renderer/humans/tools/junor_cull.py <groomed_asset_dir>
   ```
4. The skin-detail and photo-albedo inputs come from the LPS scan and from your own concept photographs; the scripts that make
   them are in `skin_kit/` (see `skin_kit/SOURCES.txt`) and `concept_fit/` (see `concept_fit/README.md`). Set `JUNOR_PHOTO` to the
   directory holding `photo_albedo.png`, `photo_w2.png`, `photo_fill.png` and `photo_meta.json`; without it the build falls back to
   the procedural albedo.
