"""EVAL: resolve a unified-system spec to (run name, bundle dir, frozen code snapshot, style).

  unified:<bundle>[@energy=0.2,mirroring=0.5,expressiveness=0.9][#tag]

<bundle> = a bundle dir (has config.json), a run dir holding bundle/, BEST (= $T/runs/BEST: a symlink, or a text file
naming a run / bundle path), a run name under $T/runs (or $EVAL_RUN_ROOTS), or `stub` (the runtime's copy-delay core).
The run name is unified-<tag>, tag = the bundle's run name (+ a style suffix), so outputs of different bundles never mix.
The unified/ code is snapshotted per bundle ($EVAL_MODEL_CODE at eval time, + its git rev) into
$E/code/<tag>/ so a later edit in that checkout never changes an evaluated result.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import os

from common import E, T, WT, MODEL_CODE, atomic_json

RUN_ROOTS = [T / "runs"] + [Path(p) for p in os.environ.get("EVAL_RUN_ROOTS", "").split(os.pathsep) if p]
STYLE_KEYS = ("energy", "mirroring", "expressiveness")


def _bundle_of(p: Path) -> Path | None:
    p = Path(p)
    if (p / "config.json").exists():
        return p
    if (p / "bundle" / "config.json").exists():
        return p / "bundle"
    return None


def read_best(best_path: Path = T / "runs/BEST") -> Path | None:
    """$T/runs/BEST -> bundle dir (symlink, or a text file whose first token is a path or run name)."""
    b = Path(best_path)
    if not b.exists() and not b.is_symlink():
        return None
    if b.is_dir():
        return _bundle_of(b.resolve())
    try:
        txt = b.read_text().strip().split()
    except OSError:
        return None
    if not txt:
        return None
    tok = txt[0]
    for cand in [Path(tok)] + [r / tok for r in RUN_ROOTS]:
        bd = _bundle_of(cand)
        if bd is not None:
            return bd.resolve()
    return None


def resolve_bundle(name: str) -> Path | None:
    if name in ("stub", "none", ""):
        return None
    if name == "BEST":
        bd = read_best()
        if bd is None:
            raise FileNotFoundError("runs/BEST does not name a bundle yet")
        return bd
    for cand in [Path(name)] + [r / name for r in RUN_ROOTS]:
        bd = _bundle_of(cand)
        if bd is not None:
            return bd.resolve()
    raise FileNotFoundError(f"no bundle at {name!r}")


def bundle_tag(bd: Path | None) -> str:
    if bd is None:
        return "stub"
    parts = [x for x in bd.parts if x not in ("bundle",)]
    # .../runs/<run>/bundle -> <run>; .../runs/<run>/<sub>/bundle -> <run>.<sub>
    try:
        i = max(i for i, x in enumerate(parts) if x == "runs")
        tag = ".".join(parts[i + 1:])
    except ValueError:
        tag = parts[-1]
    return tag.replace("/", ".")


def parse_style(s: str) -> dict:
    out = {}
    for kv in s.split(","):
        if not kv.strip():
            continue
        k, v = kv.split("=")
        k = k.strip()
        if k not in STYLE_KEYS:
            raise ValueError(f"style key {k!r} not in {STYLE_KEYS}")
        out[k] = None if v.strip().lower() in ("null", "none") else float(v)
    return out


def style_suffix(st: dict) -> str:
    return "s" + "".join(f"{k[:2]}{'n' if v is None else int(round(v * 100))}" for k, v in sorted(st.items()))


def bundle_sha(bd: Path | None) -> str:
    if bd is None:
        return "stub"
    h = hashlib.sha256()
    for f in sorted(bd.iterdir()):
        if f.is_file() and f.suffix in (".json", ".pt", ".onnx", ".npz"):
            h.update(f.name.encode())
            st = f.stat()
            h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def snapshot_code(tag: str, src: Path = MODEL_CODE) -> Path:
    """copy src/unified/*.py to $E/code/<tag>/ once; returns the snapshot root (canon1 comes from this checkout)."""
    dst = E / "code" / tag
    if (dst / "unified/runtime.py").exists():
        return dst
    for pkg in ("unified",):
        sd = src / pkg
        if not sd.is_dir():
            continue
        (dst / pkg).mkdir(parents=True, exist_ok=True)
        for f in sd.glob("*.py"):
            shutil.copy2(f, dst / pkg / f.name)
        for sub in sd.iterdir():                # small helper sub-packages
            if sub.is_dir() and not sub.name.startswith("__"):
                shutil.copytree(sub, dst / pkg / sub.name, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.npz", "*.npy"))
    rev = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "unified"], capture_output=True, text=True).stdout.strip()
    atomic_json(dst / "SNAPSHOT.json", {"src": str(src), "rev": rev, "dirty_unified": bool(dirty)})
    return dst


def resolve(spec: str) -> dict:
    """'unified:<bundle>[@style][#tag]' -> {system:'unified', run, tag, bundle, style, code}."""
    assert spec.startswith("unified")
    body = spec.split(":", 1)[1] if ":" in spec else "BEST"
    tag_override = None
    if "#" in body:
        body, tag_override = body.split("#", 1)
    style = {}
    if "@" in body:
        body, st = body.split("@", 1)
        style = parse_style(st)
    bd = resolve_bundle(body)
    base_tag = tag_override or bundle_tag(bd)
    tag = base_tag + (("-" + style_suffix(style)) if style else "")
    # code: the stub uses this checkout's unified code; trained bundles use $EVAL_MODEL_CODE (whatever it holds when
    # the bundle is first evaluated; rev + dirty recorded)
    src = WT if bd is None else MODEL_CODE
    code = snapshot_code(base_tag, src)
    meta = {"system": "unified", "run": "unified-" + tag, "tag": tag, "base_tag": base_tag,
            "bundle": None if bd is None else str(bd), "bundle_sha": bundle_sha(bd), "style": style or None,
            "code": str(code)}
    if bd is not None and (bd / "config.json").exists():
        try:
            cfg = json.loads((bd / "config.json").read_text())
            meta["bundle_kind"] = cfg.get("kind")
            meta["bundle_version"] = cfg.get("version")
        except ValueError:
            pass
    return meta
