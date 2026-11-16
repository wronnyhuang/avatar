"""render: find + load the Juno asset through ASSET's reference loader (renderer/humans/asset_io.py).

Until the asset branch is merged, the loader is imported from the asset worktree (env JUNO_ASSET_IO or the default)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                    # renderer/


def _asset_io():
    try:
        sys.path.insert(0, ROOT)
        from humans import asset_io      # noqa: F401  (merged tree)
        return asset_io
    except ImportError:
        pass
    finally:
        if sys.path and sys.path[0] == ROOT:
            sys.path.pop(0)
    alt = os.environ.get("JUNO_ASSET_IO", os.path.join(ROOT, "humans"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("juno_asset_io", os.path.join(alt, "asset_io.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_asset_dir():
    for d in (os.environ.get("JUNO_ASSET"),
              os.path.join(ROOT, "humans", "juno", "juno.asset"),
              os.path.join(ROOT, "humans", "junor", "junor.asset")):
        if d and os.path.exists(os.path.join(d, "juno.asset.json")):
            return d
    raise FileNotFoundError("no Juno asset found (set JUNO_ASSET)")


def char_asset_dir(animal):
    """pipe: humans/<animal>/<animal>.asset of a mesh character other than juno (junor); None = the default (juno's) path"""
    if animal != "juno" and os.environ.get(animal.upper() + "_ASSET"):   # JUNOR_ASSET=<dir> (offline renders of a candidate build)
        return os.environ[animal.upper() + "_ASSET"]
    d = os.path.join(ROOT, "humans", animal, animal + ".asset")
    return d if animal != "juno" and os.path.exists(os.path.join(d, "juno.asset.json")) else None


def load(asset_dir=None, lod="lod0", hair=True):
    io = _asset_io()
    d = asset_dir or default_asset_dir()
    try:
        return io.load_asset(d, lod=lod, hair=hair, validate=True)
    except ValueError as e:          # a placeholder may miss optional things: load anyway, say so once
        print(f"[mesh] asset {d}: validation: {str(e)[:300]}", file=sys.stderr)
        return io.load_asset(d, lod=lod, hair=hair, validate=False)
