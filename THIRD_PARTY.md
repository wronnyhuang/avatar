# Third-party notices

This repository is MIT-licensed (author: github user wronnyhuang). The character "Juno · realistic" (`junor`) is built on the
third-party material below. Per-file detail and rebuild steps: `renderer/humans/junor/build/LICENSES.md`.
No MetaHuman assets and no assets from games or films are used. The reference captures used by the blind judge harness in
`tools/aaa_judge` are **not** included; users supply their own (see `docs/JUNO-REALISTIC.md`).

## ICT-FaceKit (MIT)

Source: https://github.com/ICT-VGL/ICT-FaceKit. Used for the head mesh, identity model, expression shapes, teeth, gums, tongue,
mouth socket, lash cards and UV layout. The built asset contains geometry derived from it.

```
MIT License

Copyright (c) 2020 USC Institute for Creative Technologies

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Infinite, 3D Head Scan by Lee Perry-Smith (CC BY 3.0)

> "Infinite, 3D Head Scan" by Lee Perry-Smith (Infinite-Realities, www.ir-ltd.net), based on a work at www.triplegangers.com,
> licensed under CC BY 3.0. Modified: registered to ICT-FaceKit topology, band-passed, beard/brow areas replaced with synthesized
> pore detail, re-baked into ICT UV space.

- Author: Lee Perry-Smith, Infinite-Realities (http://www.ir-ltd.net/infinite-3d-head-scan-released), based on a work at www.triplegangers.com
- Source copy used: https://casual-effects.com/g3d/data10/research/model/lpshead/lpshead.zip
- Licence: Creative Commons Attribution 3.0 Unported, https://creativecommons.org/licenses/by/3.0/ . The scan's own licence note reads:
  "Permissions beyond the scope of this license may be available at http://www.ir-ltd.net/".
- Modified: yes. Only the high-frequency pore and fine-wrinkle detail is used, re-baked into the ICT-FaceKit UV space; the scan
  itself is not redistributed. The derived detail maps are part of the textures of `renderer/humans/junor/junor.asset`.

## MPFB2 / MakeHuman assets (CC0 1.0)

Source: https://github.com/makehumancommunity/mpfb2 (v2.0.18 data: hm08 base mesh, targets, `game_engine` rig and weights, two lip
and mouth masks). The MPFB2 assets are released under CC0 1.0 Universal (`LICENSE.ASSETS.md` in that repository); no attribution
is required, it is given here as a courtesy. The MPFB2 source code is GPLv3 and is not used or included. The built asset
contains a body mesh and skeleton derived from the data.

## Generated images

The face base colour (`skin_head_albedo*.jpg` in the asset textures) is projected from photographs of an invented person
generated with `google/gemini-3-pro-image` through OpenRouter. The photographs themselves are not included; only the projected,
baked texture is. Google's Gemini API terms state that "Google won't claim ownership over that content" and that the user is
responsible for the generated content and its further use; OpenRouter's terms defer ownership of outputs to the model provider's
terms (Terms of Service section 6.1). The texture is shipped on the basis that, to the best of the author's knowledge, these terms
permit it. The person depicted is invented and is not meant to resemble any real individual.

Terms: https://openrouter.ai/terms , https://ai.google.dev/gemini-api/terms (Additional Terms for the Gemini API),
https://policies.google.com/terms/generative-ai/use-policy .

## Tools used during the build (nothing shipped)

- TRELLIS.2, Microsoft, MIT, https://github.com/microsoft/TRELLIS.2: used as a shape reference when fitting the head.
- MediaPipe Face Landmarker, Google, Apache-2.0, https://github.com/google-ai-edge/mediapipe: face landmarks for the concept fit
  and for the skin-detail registration. (The live page also runs MediaPipe in the browser, see the main README.)
