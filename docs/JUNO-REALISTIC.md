# Juno · realistic (`junor`)

A photoreal mesh-rendered woman, as opposed to the SDF-ray-marched characters in `renderer/animals/`. Page character key
`junor`, menu label "Juno · realistic", URL `&char=junor`. She uses the same rig controls, face channels, voice and planner as
every other character. The classic stylized `juno` character is not part of this repository, so there is no look switch; `junor`
is selected like any other character. The neural render-to-real pass that was prototyped for her is not included.

## Pipeline

1. **Head.** The ICT-FaceKit scan-model head (MIT) is fitted to a designed face: a PCA identity blend (`build/identity.npy`)
   refined against generated concept photographs with MediaPipe landmarks and analysis-by-synthesis (`build/concept_fit/`),
   giving `build/neutral_concept.npy`. ICT's 53 expression shapes are renamed to ARKit, with combined and corrective shapes added.
2. **Body.** MPFB2 / MakeHuman base mesh, targets and `game_engine` rig (CC0), adult female macro targets, neck warped onto the
   ICT neck loop (`build/graft.py`).
3. **Skin.** Base colour is a photo albedo projected from generated photographs of an invented person (`concept_fit/proj_bake.py`,
   `finalize.py`, `brow_bake.py`). Pore, cavity and roughness detail comes from the high-frequency part of the Lee Perry-Smith head
   scan re-baked into ICT UVs (`build/skin_kit/`; CC BY 3.0, see `THIRD_PARTY.md`).
4. **Shading** (`renderer/mesh/`): PBR skin with subsurface scattering, eyes (cornea, iris, wet tearline, catchlight), mouth interior,
   cloth, soft key plus fill with a set-dressed room behind her, temporal anti-aliasing.
5. **Hair** (`groom/`, drawn by `renderer/mesh/hair.py`): strand hair, brows and lashes, drawn as true-width strands with analytic
   coverage so the silhouette stays soft. `regroom.py` regenerates the groom for any built head.
6. **Life layer** (`renderer/humans/human_life.py`, enabled by the `life` block in `junor.face.json`): blinks, micro-motion, eye
   saccades and speech-coupled movement on top of the shared face channels.

## Rebuilding the asset

The built asset is committed at `renderer/humans/junor/junor.asset` (about 130 MB, every file under 50 MB), so none of this is
needed to run her. Sources, licences and fetch instructions: `renderer/humans/junor/build/LICENSES.md`.

```
# 1. fetch ICT-FaceKit and convert it (also fetch the MPFB2 v2.0.18 data directory by hand)
renderer/humans/junor/build/get_sources.sh <work-dir>

# 2. build: positional args are <mpfb_data_dir> <ict_npz> <out_asset_dir>
python renderer/humans/junor/build/build_junor.py <mpfb_data_dir> <work-dir>/ict.npz <out_dir>/junor.asset \
    [--identity identity.npy] [--version v1] [--tex 2048]

# 3. hair groom (positional: source asset, output asset)
python renderer/humans/junor/groom/regroom.py <out_dir>/junor.asset <out_dir>/junor.groomed \
    [--n 72000] [--n1 20000] [--seed N] [--style lob] [--no-brows] [--no-lashes] [--render-cfg cfg.json]

# 4. cull triangles that can never be seen (cloth under cloth); idempotent
python renderer/humans/tools/junor_cull.py <out_dir>/junor.groomed [--reach 0.08] [--erode 3] [--dry]
```

Set `JUNOR_PHOTO` to a directory with `photo_albedo.png`, `photo_w2.png`, `photo_fill.png` and `photo_meta.json` to use your own
photo-projected albedo; without it the build uses the procedural skin colour. The scripts that make those files are in
`build/concept_fit/` (order: `README.md` there) and `build/skin_kit/`. They need MediaPipe and the Lee Perry-Smith scan, neither of
which is in this repository. Run the three groom/cull steps after every head build.

## Stills

```
python renderer/tools/juno_still.py --out x.png [--w 1920 --h 1080] [--cam face|card|bust|bust3|mid|hands|full|scene] \
    [--pose NAME] [--expr NAME] [--params shot.json|P.npy|P.json] [--asset DIR] [--lod lod0|lod1] [--device cpu|gpu]
```
`--params` takes ARKit blendshape weights plus head and gaze angles (`{"blendshapes":{...},"head_deg":{...},"gaze_deg":{...}}`) or a
rig control vector. `--no-scene` renders her alone with alpha. CPU rendering (Mesa llvmpipe) works but takes minutes at 1080p.

## Blind judge harness (`tools/aaa_judge`)

A Claude-based blind comparison of renders against reference images of shipped game characters: pairwise "which is the
shipped AAA one", a 1-10 scorecard per axis (skin, eyes, mouth/teeth, hair, lighting, facial form, clothing/neck), ranked "tells".
The reference captures are **not** included. You supply your own.

- Set `JT` to a work directory (default `tools/aaa_judge/work/`); the scripts use `$JT/judge/` for references and results and write
  logs to `$JT/logs/judge.md`.
- References go in `$JT/judge/ref/closeup/*.png` (1080x1080 crops, head about 65 % of the crop) and `$JT/judge/ref/bust/*.png`
  (1920x1080, head about 32 % of the frame), plus `$JT/judge/ref/manifest.json`:
  `{"items":[{"id":"c01","set":"closeup","group":"neutral|expression|talking","file":"<abs path to the png>"} ...]}`.
  Build the reference set yourself from captures you have the right to use for private comparison; the references are for judging
  only, do not redistribute them.
- Render the fixed shot list (`shots_aaa.json`): `python tools/aaa_judge/render_shots.py --worktree <repo> --char junor --asset <asset dir> --out $JT/judge/ours/NAME`
- Judge it: `python -I tools/aaa_judge/aaa_judge.py round --ours $JT/judge/ours/NAME --name NAME [--n 20] [--sets closeup,bust]`.
  Output: `$JT/judge/rounds/NAME/report.md`, `results.json`, `tells.json`. Judging calls the `claude` CLI (`JUDGE_CLAUDE` env var for
  its path), model `sonnet`.
- One command for both: `tools/aaa_judge/run_round.sh NAME WORKTREE ASSET_DIR [char]`.
- Bar used during development: pick-rate for ours at least 35 % in close-up and bust, and every scorecard axis within 1 point of
  the reference mean. A round of 20 pairs has about +-20 points of noise.

## Hardware

Measured with the mesh renderer at 1080p: the mesh pass takes about 12 ms on an RTX A4500-class GPU and about 22 ms on an
A4000-class GPU (hair costs about 6 ms of that on the A4000 in close-ups). At 22 ms the stream still runs near 24 fps but lip-sync
latency grew during a long turn and the adaptive quality dropped to 720p. A Quadro K620 (2 GB) renders her at 270p only, at
roughly 15 fps. Her asset adds a few hundred MB of GPU memory over the SDF characters. Textures and meshes load on first selection:
the first switch to `junor` takes tens of seconds while the set-dressing background bakes.

## Known gaps

- **Not yet AAA.** In the blind judge she was never picked over the game references (0 of 20 pairs, close-up and bust). Scorecard
  means of about 5 out of 10 against about 7 to 7.9 for the references; weakest axes are eyes and mouth, then hair and lighting.
- Hair is the main tell: strands read well up close, but the groom is flat and a pale patch shows in the part.
- The key light leaves a hard terminator across the nose and eye; the shadow side of face and neck is muddy.
- Knit sleeves smear and stretch on bent arms; small specks can show at the shoulder tops on an overhead stretch.
- The live H.264 stream softens the face and flattens hair detail at 1080p.
- Gestures that land at chest height (clap, heart) sit below the tight home framing.
- Teeth read small in a wide smile; lips can show faint streaks in some smile shapes.
- The neural render-to-real pass is not shipped (it changed her face and hair and added about 15 ms per frame).
- The rebuild needs data and tools that are not in the repository (MPFB2 data, the LPS scan, MediaPipe, your own concept photos).
