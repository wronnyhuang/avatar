"""Board brain: render a set of Mermaid diagrams to chalk-on-slate PNGs for eyeballing.

    python tools/diagram_demo.py [out_dir]

Default out_dir: <repo>/runs/diagram_demo
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "app"))

from PIL import Image, ImageDraw  # noqa: E402

from board.diagram import render_mermaid  # noqa: E402
from board.font import HandFont  # noqa: E402

DEMOS = {
    "01_decision": "graph TD; A[Start] --> B{x > 0?}; B -->|yes| C(Positive); B -->|no| D{x = 0?}; "
                   "D -->|yes| E((Zero)); D -->|no| F(Negative); C --> G([Done]); E --> G; F --> G",
    "02_water_cycle": "graph TD\n  E[Evaporation] --> C[Condensation]\n  C --> P[Precipitation]\n"
                      "  P --> L[Collection]\n  L --> E",
    "03_pipeline_lr": "flowchart LR; R[Raw data] --> C[Clean] --> F[Features] --> M[[Train model]]",
    "04_tree": "graph TD; K[Kingdom] --> P[Phylum]; K --> P2[Phylum 2]; P --> C1[Class A]; P --> C2[Class B];"
               " P2 --> C3[Class C]; C1 --> O1[Order 1]; C1 --> O2[Order 2]",
    "05_labelled_edges": "graph LR\n  S((Solid)) -- melting --> L((Liquid))\n  L -- boiling --> G((Gas))\n"
                         "  G -- condensation --> L\n  L -- freezing --> S\n  S -.->|sublimation| G",
    "06_photosynthesis": 'flowchart LR\n  CO2["Carbon dioxide CO₂"] --> P{{Photosynthesis}}\n'
                         "  W[Water H₂O] --> P\n  Sun([Sunlight]) -.-> P\n  P --> G[Glucose]\n  P --> O[Oxygen O₂]",
    "07_timeline": "timeline\n  title The Space Race\n  1957 : Sputnik 1\n  1961 : Gagarin orbits Earth\n"
                   "       : Freedom 7\n  1969 : Apollo 11 Moon landing\n  1975 : Apollo-Soyuz",
    "08_mindmap": "mindmap\n  root((Plants))\n    Roots\n      Absorb water\n      Anchor\n    Leaves\n"
                  "      Photosynthesis\n      Transpiration\n    Stem\n      Transport\n    Flowers\n"
                  "      Pollination\n      Seeds",
    "09_malformed": "graph TD; A[Start --> B{Check; B -->|ok| C(Go; C --> D[[Fin",
    "10_math": "graph TD; Q[\"$ax^2+bx+c=0$\"] --> D{\"$\\Delta = b^2-4ac$\"}; "
               "D -->|\"$\\Delta>0$\"| R2[two real roots]; D -->|\"$\\Delta=0$\"| R1[one root]; "
               "D -->|\"$\\Delta<0$\"| R0[no real roots]; R2 --> F[\"$x=\\frac{-b\\pm\\sqrt{\\Delta}}{2a}$\"]",
    "11_shapes_subgraph": "flowchart TB\n  subgraph inputs [Inputs]\n    a[/Read file/]\n    b[(Database)]\n  end\n"
                          "  a --> c>Parse] & d{{Validate}}\n  b --> d\n  c ==> e[\\Report\\]\n  d --> e\n"
                          "  e --> e2((( End )))\n  classDef x fill:#f00\n  class a x",
    "12_food_chain_bt": "graph BT; Sun --> Grass; Grass --> Rabbit; Grass --> Grasshopper; Grasshopper --> Frog; "
                        "Rabbit --> Fox; Frog --> Snake; Snake --> Hawk; Fox --> Hawk",
    # dense labelled decisions (vision-judge repro first); rendered for a wide 1.76 x 0.62 board region
    "13_rain_repro": "graph TD; A[Rain forecast?] -->|No| B[Look at sky]; A -->|Yes| C[Bring umbrella]; "
                     "B -->|Dark clouds| C; B -->|Clear| D[Leave it home]",
    "14_dense_td": "graph TD; S{Is the number even?} -->|yes| H[Divide by 2]; S -->|no| T[Multiply by 3, add 1]; "
                   "H -->|result is 1| E([Stop]); H -->|otherwise| S; T -->|always| S; T -->|overflow| X[Error]",
    "15_dense_lr": "flowchart LR; Q{Which state?} -->|below 0 C| I[Ice]; Q -->|0 to 100 C| W[Water]; "
                   "Q -->|above 100 C| V[Steam]; I -->|melts| W; W -->|boils| V; V -->|condenses| W",
    "16_dense_td2": "graph TD; G{Grade check} -->|90 and up| A[A]; G -->|80 to 89| B[B]; G -->|70 to 79| C[C]; "
                    "G -->|below 70| R[Retake test]; R -->|pass| C; R -->|fail| T[Tutoring]; T -->|try again| R",
}
REGION = {k: (1.76 / 0.072, 0.62 / 0.072) for k in ("13_rain_repro", "14_dense_td", "15_dense_lr", "16_dense_td2")}


def draw(res, path, W=1024, H=512, margin=28):
    img = Image.new("RGB", (W, H), (38, 52, 48))
    d = ImageDraw.Draw(img)
    w, h = max(res["w"], 1e-6), max(res["h"], 1e-6)
    k = min((W - 2 * margin) / w, (H - 2 * margin) / h)
    ox, oy = (W - w * k) / 2, (H - h * k) / 2
    for st in res["strokes"]:
        pts = [(ox + x * k, oy + y * k) for x, y in st]
        if len(pts) >= 2:
            d.line(pts, fill=(236, 236, 228), width=3, joint="curve")
    img.save(path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "diagram_demo")
    os.makedirs(out, exist_ok=True)
    font = HandFont(seed=7)
    for name, src in DEMOS.items():
        t = time.perf_counter()
        mw, mh = REGION.get(name, (2.0, 1.0))
        res = render_mermaid(src, font, 1.0, max_w=mw, max_h=mh)
        dt = (time.perf_counter() - t) * 1000
        p = os.path.join(out, name + ".png")
        draw(res, p)
        npts = sum(len(s) for s in res["strokes"])
        print(f"{name:22s} {dt:6.1f} ms  {len(res['nodes']):2d} nodes {len(res['strokes']):4d} strokes {npts:6d} pts "
              f"w={res['w']:.1f} h={res['h']:.1f}  {res['warnings']}")


if __name__ == "__main__":
    main()
