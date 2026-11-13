#!/usr/bin/env python3
"""objects-eval — step 3: key frames of a case's main segment through the candidate's renderer -> strip PNG.

Keys: the main segment's start (0.3 s before his first main turn = the "before" state), every object / kf / move event
start+end, every attachment change of a prop (picked up / let go / put on / taken off: prop{k}_hl|hr|hh flips), the
moment a free prop stops moving after flight (landed), then filled evenly to N, always ending on the last frame.
Camera: the page's own cam columns, with a zoom FLOOR per key so every relevant free prop (mentioned in his words or moved
in this case) and the muse fit in the frame (calibrated: zoom z shows ~±(1.4 + 1.06 z) H across, up to ~1.2 + 1.0 z H).
"""
import hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

from checks import Scene

EVAL = Path(os.environ.get("OBJECTS_EVAL_DIR") or Path(__file__).resolve().parents[2] / "runs" / "objects_eval")   # workspace: candidate/, runs/, cache/
HERE = Path(__file__).resolve().parent
PY_LIVE = os.environ.get("PYTHON") or sys.executable   # interpreter with the renderer's dependencies
W, H = 640, 360


def relevant_props(S, texts, f0):
    words = " ".join(texts).lower()
    out = set()
    for pid, p in S.props.items():
        names = [pid.replace("_", " "), p.get("name", "")] + list(p.get("aliases") or [])
        if any(n and re.search(r"\b" + re.escape(n.lower()) + r"\b", words) for n in names):
            out.add(pid)
    for pid in S.props:
        k = S.slot(pid)
        for f in range(f0, len(S.rows), 5):
            if S.attached(pid, f) or any(abs(v) > 0.02 for v in S.offset(pid, f)[:3]):
                out.add(pid); break
    return sorted(out)


def zoom_for(S, f, rel):
    cx = S.col("cam_x", f)
    need_x, need_y = abs(S.col("root_x", f) - cx) + 0.45, 1.0
    for pid in rel:
        Wp = S.world(pid, f)
        if Wp is None or pid in ("desk",):
            continue
        need_x = max(need_x, abs(Wp[0] - cx) + 0.2)
        need_y = max(need_y, Wp[1] + 0.15)
    z = max((need_x - 1.4) / 1.06, need_y - 1.2, 0.0)
    if abs(S.col("root_x", f)) > 0.4 or abs(S.col("root_z", f)) > 0.4:
        z = max(z, 0.22)        # the body walked off its spot: a little wider for the room context
    return round(min(1.0, z + (0.08 if z > 0 else 0)), 2)


def pick_keys(S, case, n=12):
    f0 = max(0, case["main_frame0"] - 9)
    fN = len(S.rows) - 1
    fps = 30
    marks = set()
    for e in case.get("events") or []:
        if e.get("kind") in ("move", "object", "kf") and e.get("phase") in ("start", "end"):
            fr = int(round(e["t"] * fps)) - 1
            if f0 <= fr <= fN:
                marks.add(min(fN, fr + (3 if e["phase"] == "start" else -2)))
    for pid in S.props:
        prev = None
        for f in range(f0, fN + 1):
            a = (S.held(pid, f), S.on_head(pid, f))
            if prev is not None and a != prev:
                marks.add(max(f0, f - 1)); marks.add(min(fN, f + 4))
            prev = a
    marks = sorted(m for m in marks if f0 < m < fN)
    ks = [f0] + marks + [fN]
    while len(ks) < n:
        gaps = sorted([(b - a, a, b) for a, b in zip(ks, ks[1:])], reverse=True)
        g, a, b = gaps[0]
        if g < 8:
            break
        ks = sorted(ks + [(a + b) // 2])
    if len(ks) > n:
        idx = sorted({round(i * (len(ks) - 1) / (n - 1)) for i in range(n)})
        ks = [ks[i] for i in idx]
    return ks


def label(S, f, st):
    bits = []
    for pid in S.props:
        if S.held(pid, f):
            bits.append(f"holds {pid}")
        elif S.on_head(pid, f):
            bits.append(f"head:{pid}")
    if st:
        if st.get("posture") and st["posture"] != "standing":
            bits.insert(0, st["posture"])
        if st.get("activity"):
            bits.append(str(st["activity"]).split(" (")[0])
    return ", ".join(bits)


def make_strip(src, case_dir, case, n=12, threads=12, cols=4):
    case_dir = Path(case_dir)
    rig = case_dir / "rig.jsonl"
    S = Scene(src, rig, case.get("state"))
    keys = pick_keys(S, case, n)
    rel = relevant_props(S, list(case.get("setup") or []) + list(case.get("turns_text") or []), case["main_frame0"])
    zooms = [zoom_for(S, f, rel) for f in keys]
    states = json.loads((case_dir / "rig.states.json").read_text()) if (case_dir / "rig.states.json").exists() else []
    th = hashlib.sha256()
    for p in sorted((Path(src) / "renderer").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            th.update(str(p).encode()); th.update(str(p.stat().st_size).encode()); th.update(str(int(p.stat().st_mtime)).encode())
    key = hashlib.sha256((rig.read_text() + th.hexdigest() + json.dumps([keys, zooms, W, H])).encode()).hexdigest()[:20]
    cache = EVAL / "cache/frames" / key
    if not (cache / "done").exists():
        shutil.rmtree(cache, ignore_errors=True); cache.mkdir(parents=True)
        env = dict(os.environ, AVATAR_RT_DEVICE="cpu", LP_NUM_THREADS=str(threads), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        cmd = ["nice", "-n", "10", PY_LIVE, "-B", str(HERE / "render_keys.py"), "--renderer", str(Path(src) / "renderer"),
               "--rig", str(rig), "--keys", ",".join(f"{f / 30:.4f}" for f in keys), "--zoom", ",".join(map(str, zooms)),
               "--out-dir", str(cache), "--w", str(W), "--h", str(H), "--threads", str(threads)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        (cache / "render.log").write_text(r.stdout[-3000:] + "\n" + r.stderr[-6000:])
        if r.returncode != 0:
            raise RuntimeError(f"render failed: {r.stderr[-600:]}")
        (cache / "done").write_text("ok")
    t0 = case["main_frame0"]
    words = " / ".join(case.get("turns_text") or [])
    pre = ("setup: " + " / ".join(case.get("setup") or []) + "   ") if case.get("setup") else ""
    states_at = lambda f: states[f] if f < len(states) else None  # noqa: E731

    def sheet_of(idx, cols_, out, part=""):
        c = min(cols_, len(idx)); rn = (len(idx) + c - 1) // c; lh = 18; top = 40
        sheet = Image.new("RGB", (c * W, rn * (H + lh) + top), (250, 250, 250)); d = ImageDraw.Draw(sheet)
        d.text((6, 4), f"{case['id']} [{case.get('kind')}] {part} {pre}"[:300], fill=(90, 90, 90))
        d.text((6, 20), f"HE SAYS: \"{words}\""[:300], fill=(0, 0, 0))
        for j, i in enumerate(idx):
            f = keys[i]
            x, y = (j % c) * W, top + (j // c) * (H + lh)
            im = cache / f"k{i:02d}.png"
            if im.exists():
                sheet.paste(Image.open(im).convert("RGB"), (x, y + lh))
            d.text((x + 4, y + 3), (f"#{i+1} t={(f - t0) / 30:+.1f}s z{zooms[i]:.2f}  " + label(S, f, states_at(f)))[:90], fill=(20, 20, 160))
        sheet.save(out)
        return str(out)
    allk = list(range(len(keys)))
    out = Path(sheet_of(allk, cols, case_dir / "strip.png"))
    half = (len(keys) + 1) // 2
    parts = [sheet_of(allk[:half], 3, case_dir / "strip_a.png", "(frames 1-%d)" % half)]
    if len(keys) > half:
        parts.append(sheet_of(allk[half:], 3, case_dir / "strip_b.png", "(frames %d-%d)" % (half + 1, len(keys))))
    for i in allk:
        if (cache / f"k{i:02d}.png").exists():
            shutil.copyfile(cache / f"k{i:02d}.png", case_dir / f"k{i:02d}.png")
    return {"strip": str(out), "parts": parts, "keys": keys, "zooms": zooms, "relevant": rel}
