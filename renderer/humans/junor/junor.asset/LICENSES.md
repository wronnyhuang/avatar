# junor.asset: sources and licences

| Part | Source | Licence | What was changed |
|---|---|---|---|
| Head geometry, identity, expression shapes, teeth, gums/tongue, mouth socket, lash cards, UV layout | ICT-FaceKit, USC Institute for Creative Technologies, https://github.com/ICT-VGL/ICT-FaceKit | MIT, Copyright (c) 2020 USC Institute for Creative Technologies | Identity blended from PCA modes; expressions renamed to ARKit, combined and corrected; grafted onto the body at the neck loop |
| Body, skeleton, skin weights | MPFB2 / MakeHuman hm08 base, targets, `game_engine` rig, https://github.com/makehumancommunity/mpfb2 (data only) | CC0 1.0 Universal | Adult female targets, neck warped onto the ICT neck loop |
| Eyes, wardrobe, cloth textures, procedural skin | Original code of this repository | Same licence as this repository | none |
| Skin pore / cavity / roughness / albedo detail (head) | "Infinite, 3D Head Scan" by Lee Perry-Smith (Infinite-Realities, www.ir-ltd.net), based on a work at www.triplegangers.com, https://casual-effects.com/g3d/data10/research/model/lpshead/lpshead.zip | CC BY 3.0 Unported | Modified: registered to ICT-FaceKit topology, band-passed, beard/brow areas replaced with synthesized pore detail, re-baked into ICT UV space |
| Face base colour (`tex/skin_head_albedo*.jpg`) | Projected from AI-generated photographs of an invented person (google/gemini-3-pro-image via OpenRouter) | Generated content, see THIRD_PARTY.md | Projected onto the ICT UV atlas |

Attribution text required by CC BY 3.0:

> "Infinite, 3D Head Scan" by Lee Perry-Smith (Infinite-Realities, www.ir-ltd.net), based on a work at www.triplegangers.com,
> licensed under CC BY 3.0. Modified: registered to ICT-FaceKit topology, band-passed, beard/brow areas replaced with synthesized
> pore detail, re-baked into ICT UV space.

No MetaHuman or game assets are used. Full notices: `THIRD_PARTY.md` at the repository root. Source table with rebuild steps:
`renderer/humans/junor/build/LICENSES.md`.
