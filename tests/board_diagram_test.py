"""boardbrain: tests for app/board/diagram.py (Mermaid -> hand-traceable polylines).

Run:  python tests/board_diagram_test.py      (plain asserts)
  or  pytest tests/board_diagram_test.py
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app"))

from board.diagram import render_mermaid  # noqa: E402
from board.font import HandFont  # noqa: E402
from board import geom  # noqa: E402

FONT = HandFont(seed=3)

GOOD = [
    "graph TD; A[Start] --> B{x > 0?}; B -->|yes| C(Positive); B -->|no| D((Zero))",
    "flowchart LR\n  a --> b --> c --> d",
    "graph TD\n  E[Evaporation] --> C[Condensation]\n  C --> P[Precipitation]\n  P --> L[Collection]\n  L --> E",
    "graph TD; K --> P; K --> Q; P --> C1; P --> C2; Q --> C3; C1 --> O1; C1 --> O2",
    "graph LR; S((Solid)) -- melting --> L((Liquid)); L -- boiling --> G((Gas)); G -- condensation --> L; "
    "L -- freezing --> S; S -.->|sublimation| G",
    "graph TD; a & b --> c & d; c ==> e; d -.- e; e --- f; f --o g; g --x h; h <--> i; i ~~~ j",
    "graph BT; A([stadium]) --> B[[sub]] --> C[(db)] --> D>asym] --> E{{hex}} --> F[/para/] --> G[\\alt\\] "
    "--> H[/trap\\] --> I[\\trap2/] --> J(((dbl)))",
    'graph TD; A["quoted (with) [brackets]"] --> B["line one<br>line two"]',
    "graph TD; Q[\"$ax^2+bx+c=0$\"] --> R{\"$\\Delta>0$?\"}",
    "flowchart TB\n subgraph one [Group One]\n a1 --> a2\n end\n subgraph two\n b1 --> b2\n end\n a2 --> b1",
    "graph TD; A --> A; A --> B; B --> A",
    "timeline\n title History\n 2002 : LinkedIn\n 2004 : Facebook\n      : Google\n section Later\n 2005 : YouTube",
    "mindmap\n  root((Cells))\n    Plant\n      Wall\n      Chloroplast\n    Animal\n      Centriole",
    "stateDiagram-v2\n [*] --> Still\n Still --> Moving : push\n Moving --> Still\n Moving --> Crash\n Crash --> [*]",
    "sequenceDiagram\n participant A as Alice\n A->>B: Hello\n B-->>A: Hi",
    "```mermaid\ngraph LR\nA-->B %% comment\nclassDef big fill:#f9f\nstyle A fill:#f00\nclass A big\nlinkStyle 0 stroke:red\n```",
]

MALFORMED = [
    "",
    "   ",
    "graph",
    "graph TD;",
    "graph TD; A-->",
    "graph TD; A[Start --> B{Check; B -->|ok| C(Go; C --> D[[Fin",
    "graph TD; A[unclosed",
    "graph TD; A{{{{ --> ]]]] B",
    "graph TD; A -->|never closed B",
    "graph TD; A -- text without end",
    "flowchart LR; A --> B; B --> C; C --> ",
    "graph TD; ;;; --> --> -->",
    "graph TD; A[\"unterminated quote] --> B",
    "graph TD; A[$\\frac{1}{$] --> B[$x^{$]",
    "graph TD; A((( ))) --> B[] --> C()",
    "sequenceDiagram\nAlice->>",
    "pie title Pets\n \"Dogs\" : 386",
    "mindmap",
    "mindmap; root; a; b; c",
    "timeline",
    "timeline\n : orphan event",
    "graph TD\n subgraph s1\n a --> b",
    "A --> B --> C",
    "graph XY; A-->B",
    "graph TD; 中文 --> 日本",
    "graph TD; A@{ shape: diamond, label: \"Decide\" } --> B",
    "%%{init: {'theme':'dark'}}%%\ngraph TD; A-->B",
    "graph TD; " + "; ".join(f"n{i} --> n{i + 1}" for i in range(60)),
    "graph TD; a-->b; b-->c; c-->a; a-->c; c-->b; b-->a",
    "graph TD; A[" + "x" * 400 + "] --> B",
]


def _check(res, src=""):
    assert isinstance(res, dict), src
    for k in ("strokes", "w", "h", "nodes", "warnings"):
        assert k in res, (k, src)
    assert math.isfinite(res["w"]) and math.isfinite(res["h"]), src
    for st in res["strokes"]:
        assert len(st) >= 2, src
        for x, y in st:
            assert math.isfinite(x) and math.isfinite(y), src
            assert -1e-6 <= x <= res["w"] + 1e-6 and -1e-6 <= y <= res["h"] + 1e-6, (src, x, y, res["w"], res["h"])
    for r in res["nodes"].values():
        assert all(math.isfinite(v) for v in r), src
        assert r[0] <= r[2] and r[1] <= r[3], src
    assert not any(w.startswith("internal error") for w in res["warnings"]), (src, res["warnings"])


def test_good_inputs_render():
    for src in GOOD:
        res = render_mermaid(src, FONT, 1.0)
        _check(res, src)
        assert res["strokes"], src


def test_malformed_never_raises():
    for src in MALFORMED:
        for mw, mh in ((None, None), (2.0, 1.0), (1.0, 2.0)):
            res = render_mermaid(src, FONT, 1.0, mw, mh)
            _check(res, src)
    # malformed inputs say so
    for src in ("graph TD; A-->", "graph TD; A[unclosed", "graph TD; A -->|never closed B", ""):
        assert render_mermaid(src, FONT, 1.0)["warnings"], src


def test_partial_input_keeps_what_parses():
    res = render_mermaid("graph TD; A[Start --> B{Check; B -->|ok| C(Go; C --> D[[Fin", FONT, 1.0)
    assert set(res["nodes"]) >= {"A", "B", "C", "D"}, res["nodes"]
    res = render_mermaid("graph TD; A-->B; B-->", FONT, 1.0)
    assert set(res["nodes"]) == {"A", "B"}


def test_no_node_overlaps():
    for src in GOOD + MALFORMED:
        if src.lstrip().startswith(("timeline", "mindmap")):
            continue
        for mw, mh in ((None, None), (2.0, 1.0), (1.0, 1.5)):
            res = render_mermaid(src, FONT, 1.0, mw, mh)
            rs = list(res["nodes"].items())
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    assert not geom.rects_overlap(rs[i][1], rs[j][1], -1e-6), (src, rs[i][0], rs[j][0])


def test_mindmap_and_timeline_no_overlap():
    for src in GOOD:
        if src.lstrip().startswith(("timeline", "mindmap")):
            res = render_mermaid(src, FONT, 1.0, 2.0, 1.0)
            rs = list(res["nodes"].values())
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    assert not geom.rects_overlap(rs[i], rs[j], -1e-6), src


def test_determinism():
    for src in GOOD[:6] + MALFORMED[5:8]:
        a = render_mermaid(src, HandFont(seed=5), 1.0, 2.0, 1.0)
        b = render_mermaid(src, HandFont(seed=5), 1.0, 2.0, 1.0)
        assert a == b, src


def test_units_scale_with_text_size():
    src = GOOD[0]
    a = render_mermaid(src, FONT, 1.0)
    b = render_mermaid(src, FONT, 10.0)
    assert abs(b["w"] / a["w"] - 10.0) < 0.05 and abs(b["h"] / a["h"] - 10.0) < 0.05


def test_resampled_spacing():
    res = render_mermaid(GOOD[0], FONT, 1.0)
    worst = max(geom.seglen(st[i - 1], st[i]) for st in res["strokes"] for i in range(1, len(st)))
    assert worst <= 0.06 + 1e-6, worst


def test_parse_details():
    res = render_mermaid(GOOD[0], FONT, 1.0)
    assert set(res["nodes"]) == {"A", "B", "C", "D"}
    # TD: A above B above C/D
    n = res["nodes"]
    assert n["A"][3] < n["B"][1] < n["B"][3] < n["C"][1]
    res = render_mermaid("flowchart LR\n  a --> b --> c --> d", FONT, 1.0)
    n = res["nodes"]
    assert n["a"][2] < n["b"][0] and n["b"][2] < n["c"][0] and n["c"][2] < n["d"][0]
    res = render_mermaid("graph RL; a --> b", FONT, 1.0)
    assert res["nodes"]["b"][2] < res["nodes"]["a"][0]
    res = render_mermaid("graph BT; a --> b", FONT, 1.0)
    assert res["nodes"]["b"][3] < res["nodes"]["a"][1]


def test_aspect_switch():
    long_td = "graph TD; " + "; ".join(f"n{i}[Step {i}] --> n{i + 1}[Step {i + 1}]" for i in range(7))
    res = render_mermaid(long_td, FONT, 1.0, max_w=4.0, max_h=1.0)
    assert any("switched" in w for w in res["warnings"]), res["warnings"]
    assert res["w"] > res["h"]
    res = render_mermaid(long_td, FONT, 1.0, max_w=1.0, max_h=4.0)
    assert not any("switched" in w for w in res["warnings"])


DENSE = [
    # vision-judge repro (caller region 1.76 x 0.62 at text size 0.072)
    "graph TD; A[Rain forecast?] -->|No| B[Look at sky]; A -->|Yes| C[Bring umbrella]; B -->|Dark clouds| C; "
    "B -->|Clear| D[Leave it home]",
    "graph TD; S{Is the number even?} -->|yes| H[Divide by 2]; S -->|no| T[Multiply by 3, add 1]; "
    "H -->|result is 1| E([Stop]); H -->|otherwise| S; T -->|always| S; T -->|overflow| X[Error]",
    "flowchart LR; Q{Which state?} -->|below 0 C| I[Ice]; Q -->|0 to 100 C| W[Water]; Q -->|above 100 C| V[Steam]; "
    "I -->|melts| W; W -->|boils| V; V -->|condenses| W",
    "graph TD; G{Grade check} -->|90 and up| A[A]; G -->|80 to 89| B[B]; G -->|70 to 79| C[C]; "
    "G -->|below 70| R[Retake test]; R -->|pass| C; R -->|fail| T[Tutoring]; T -->|try again| R",
]
REGIONS = [(1.76 / 0.072, 0.62 / 0.072), (None, None), (14.0, 5.3), (6.0, 8.0)]


def _label_violations(res, clear=0.3):
    bad = []
    labs = res["edge_labels"]
    for i, r in enumerate(labs):
        for nid, o in res["nodes"].items():
            if geom.rects_overlap(r, o, clear - 1e-6):
                bad.append(f"label {i} within {clear} of node {nid}")
        for j in range(i + 1, len(labs)):
            if geom.rects_overlap(r, labs[j], 0.0):
                bad.append(f"label {i} overlaps label {j}")
    return bad


def _margin_ok(res, m=0.3):
    for r in list(res["nodes"].values()) + res["edge_labels"]:
        if r[0] < m - 1e-6 or r[1] < m - 1e-6 or r[2] > res["w"] - m + 1e-6 or r[3] > res["h"] - m + 1e-6:
            return False
    return True


def test_edge_labels_never_overlap_nodes_or_labels():
    srcs = DENSE + [s for s in GOOD if "|" in s or "-- " in s]
    for src in srcs:
        for mw, mh in REGIONS:
            res = render_mermaid(src, FONT, 1.0, mw, mh)
            _check(res, src)
            if src in DENSE:
                assert len(res["edge_labels"]) == src.count("|") // 2, src
            v = _label_violations(res)
            assert not v, (src, mw, mh, v)


def test_dense_repro_labels_clear_of_edges():
    """The 'Dark clouds' repro: every label has a spot that covers no edge line."""
    res = render_mermaid(DENSE[0], FONT, 1.0, *REGIONS[0])
    assert len(res["edge_labels"]) == 4 and not _label_violations(res)
    assert not res["warnings"], res["warnings"]


def _label_line_crossings(res, shrink=0.2):
    """Labels whose interior (shrunk by `shrink`) is crossed by a stroke that is not part of that label."""
    out = []
    for i, r in enumerate(res["edge_labels"]):
        for st in res["strokes"]:
            b = geom.bbox([st])
            if b[0] >= r[0] - 0.2 and b[2] <= r[2] + 0.2 and b[1] >= r[1] - 0.2 and b[3] <= r[3] + 0.2:
                continue      # the label's own glyph strokes
            if any(r[0] + shrink < x < r[2] - shrink and r[1] + shrink < y < r[3] - shrink for x, y in st):
                out.append(i)
                break
    return out


def test_dense_labels_cross_no_lines():
    for src in DENSE:
        for mw, mh in REGIONS:
            res = render_mermaid(src, FONT, 1.0, mw, mh)
            assert not _label_line_crossings(res), (src, mw, mh, _label_line_crossings(res))


def test_outer_margin():
    for src in GOOD + DENSE:
        for mw, mh in REGIONS[:2]:
            res = render_mermaid(src, FONT, 1.0, mw, mh)
            assert _margin_ok(res), (src, mw, mh)
            for st in res["strokes"]:
                for x, y in st:
                    assert 0.3 - 1e-6 <= x <= res["w"] - 0.3 + 1e-6 and 0.3 - 1e-6 <= y <= res["h"] - 0.3 + 1e-6, src


def test_timing_10_nodes():
    src = ("graph TD; A[Start] --> B{Is it raining?}; B -->|yes| C[Take umbrella]; B -->|no| D[Wear sunglasses]; "
           "C --> E[Walk to school]; D --> E; E --> F{Late?}; F -->|yes| G[Run]; F -->|no| H[Stroll]; "
           "G --> I[Class]; H --> I; I --> J([End])")
    render_mermaid(src, FONT, 1.0)
    best_u = best_c = 1e9
    for _ in range(5):
        t = time.perf_counter()
        r = render_mermaid(src, FONT, 1.0)
        best_u = min(best_u, time.perf_counter() - t)
        t = time.perf_counter()
        render_mermaid(src, FONT, 1.0, 2.0, 1.0)
        best_c = min(best_c, time.perf_counter() - t)
    assert len(r["nodes"]) == 10
    print(f"  10-node flowchart: {best_u * 1000:.1f} ms unconstrained, {best_c * 1000:.1f} ms with max_w/max_h")
    assert best_u < 0.05, best_u


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
