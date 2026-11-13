"""scaling-sweep packs: NESTED session-level increment packs + the whisper_tiny join (full_v4).

    base   python -I tools/train/p2_sweep_pack.py base --pack $P/packs/full_v3.qpack.npz --out-dir $P/packs/sweep \
               --name fv4 [--fracs 25,50,100] [--seed 0] [--no-lean] [--zstd 12]
    enc    python -I tools/train/p2_sweep_pack.py enc --out-dir $P/packs/sweep --name fv4 --streams $P/streams_full.jsonl \
               [--enc whisper_tiny] [--feats-dir $P/feats] [--min-coverage 0.98] [--ext $P/packs/full_v4.enc.npz]
    watch  python -I tools/train/p2_sweep_pack.py watch <enc args> [--pids-file $P/audio_work/pids_wt_full.txt] [--poll 60]
           (waits until every stream has its feature file, or the extraction pids are gone with >= --min-coverage,
            then runs `enc`; nohup it at nice 19)

INCREMENTS (base): TRAIN sessions of each dataset are shuffled (seed) and cut at --fracs percent of that dataset's
train frames (stratified: every increment has the same Seamless/L2L/Panoptic mix); increment k = the sessions between
frac k-1 and frac k. ALL val/test streams go into the FIRST increment. A lane of fraction f trains on the CONCATENATION
of increments 1..k (unified.data.ConcatPack), so 25 c 50 c 100 by construction and every frame is stored once.
<name>_inc<f>.npz = a qpack with the input's members sliced by whole streams (delta planes stay valid: every stream
starts absolute), streams re-based, partners remapped, session ids kept, plus `src_streams` (original stream index)
and names.hours / pack_meta.increment. --lean (default): a_q mel rows 0..39 zeroed (enc runs read only loud_db/vad;
zeros compress to nothing) and `lean_no_mel` = 1 (the Sampler refuses audio_in "mel" on it).
ENCODER FILES (enc): <name>_inc<f>.enc.npz = enc_<e> int8 (n,D) + enc_<e>_scale/_qmean (one scale for all
increments: 99.99th pct of |x - mean| / 127 per dim over train frames) + enc_<e>_ok (S,) per stream (0 = no feature
file: L2L/Panoptic, failed streams -> the Sampler's enc_missing rule) + enc_<e>_lookahead + enc_for (json). A lane's
pack entry is the pair [<name>_inc<f>.npz, <name>_inc<f>.enc.npz] (member-merged, see unified.data.open_pack).
--ext also writes the join for the WHOLE pack in full_v3 frame order (full_v4 = full_v3.qpack.npz +
full_v4.enc.npz as pack_extra). Every output gets a .zst (zstd -T8) and a sha256; manifest <name>_sweep.json; READY
lines in $P/packs/READY. Threads: zstd -T8, numpy single-threaded; run at nice 19.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import slim_pack  # noqa: E402  (Pack: memory-mapped members)

sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from canon1 import paths as _paths  # noqa: E402

P = str(_paths.P2)          # $MM_WORK_DIR/p2
PLANES = ("m_lo", "m_hi", "a_q", "mask")           # (C, N) column planes
FRAME = ("wexp", "style", "events", "a2f", "a2f_u8", "a2f_mask")    # (N, ...) rows
STREAM = ("quality", "events_ok")                   # (S,)


class Src:
    """an .npz: big stored members memory-mapped (slim_pack.Pack), small ones through np.load."""

    def __init__(self, path):
        self.pk = slim_pack.Pack(path)
        self.z = np.load(path, allow_pickle=False)
        self.infos = self.pk.infos

    def get(self, k):
        return self.pk[k] if self.infos[k].file_size > (64 << 20) else self.z[k]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def zst(path, level):
    t0 = time.time()
    tmp = path + ".zst.tmp"
    subprocess.run(["nice", "-n", "19", "zstd", "-q", "-f", "-T8", f"-{level}", "--long=27", path, "-o", tmp], check=True)
    os.replace(tmp, path + ".zst")
    with open(path + ".zst.done", "w") as fh:                # complete marker (consumers wait for it)
        fh.write(dt.datetime.now().astimezone().isoformat(timespec="seconds") + "\n")
    out = dict(file=os.path.basename(path), mb=round(os.path.getsize(path) / 1e6, 1),
               zst=os.path.basename(path) + ".zst", zst_mb=round(os.path.getsize(path + ".zst") / 1e6, 1),
               zst_sha256=sha256(path + ".zst"), zst_s=round(time.time() - t0, 1))
    log(f"  zst {out['zst']}: {out['mb']} -> {out['zst_mb']} MB in {out['zst_s']}s")
    return out


def savez_atomic(path, arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def ready(name, path, what, extra=None):
    line = dict(t=dt.datetime.now().astimezone().isoformat(timespec="seconds"), pack=name, what=what, path=path,
                zst=path + ".zst" if os.path.exists(path + ".zst") else None,
                mb=round(os.path.getsize(path) / 1e6), **(extra or {}))
    with open(os.path.join(P, "packs", "READY"), "a") as f:
        f.write(json.dumps(line) + "\n")


def fracs_of(s):
    f = [float(x) for x in str(s).split(",") if x]
    assert f and f[0] > 0 and all(a < b for a, b in zip(f, f[1:])) and f[-1] <= 100, f
    return f


def assign(st, fracs, seed):
    """increment id per stream (-1 = not in any listed increment): stratified per dataset over TRAIN sessions."""
    inc = np.full(len(st), -1, np.int64)
    sess, split, ds = st[:, 6], st[:, 3], st[:, 4]
    s_split = {}
    for i in range(len(st)):
        s_split.setdefault(int(sess[i]), set()).add(int(split[i]))
    mixed = [s for s, v in s_split.items() if len(v) > 1]
    assert not mixed, f"sessions spanning splits: {mixed[:5]}"
    rng = np.random.default_rng(seed)
    for d in np.unique(ds):
        tr = (ds == d) & (split == 0)
        ss = np.unique(sess[tr])
        fr = {}
        for i in np.nonzero(tr)[0]:
            fr[int(sess[i])] = fr.get(int(sess[i]), 0) + int(st[i, 1])
        rng.shuffle(ss)
        tot, acc, k = sum(fr.values()), 0, 0
        lab = {}
        for s_ in ss:
            while k < len(fracs) and acc >= fracs[k] / 100 * tot - 1e-9:
                k += 1
            lab[int(s_)] = k if k < len(fracs) else -1
            acc += fr[int(s_)]
        for i in np.nonzero(tr)[0]:
            inc[i] = lab[int(sess[i])]
    inc[split != 0] = 0                              # every val/test stream lives in the first increment
    return inc


def cmd_base(a):
    t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    pk = Src(a.pack)
    st = np.asarray(pk.get("streams"))
    N = int((st[:, 0] + st[:, 1]).max())
    fracs = fracs_of(a.fracs)
    inc = assign(st, fracs, a.seed)
    if a.train_only:                                 # an EXTRA corpus (CANDOR): the sweep's val set stays the base pack's
        inc[st[:, 3] != 0] = -1
    names = json.loads(str(pk.get("names")))
    dsets = names.get("datasets", [])
    meta0 = json.loads(str(pk.get("pack_meta"))) if "pack_meta" in pk.infos else {}
    globals_ = [k for k in pk.infos if k not in PLANES + FRAME + STREAM + ("streams", "names", "pack_meta")
                and not k.startswith("enc_")]
    encs = [k for k in pk.infos if k.startswith("enc_") and not k.endswith(("_ok",) + slim_pack.DERIVED + ("_lookahead",))]
    planes = {k: pk.get(k) for k in PLANES if k in pk.infos}         # memmaps when stored
    manifest = dict(source=os.path.abspath(a.pack), fracs=fracs, seed=a.seed, lean=a.lean, built=None, increments=[])
    for k, f in enumerate(fracs):
        sids = np.nonzero(inc == k)[0]
        if not len(sids):
            log(f"increment {f}: empty, skipped")
            continue
        idx = np.concatenate([np.arange(st[i, 0], st[i, 0] + st[i, 1]) for i in sids])
        new_id = {int(o): j for j, o in enumerate(sids)}
        ns = st[sids].copy()
        ns[:, 0] = np.concatenate([[0], np.cumsum(ns[:-1, 1])])
        bad = [int(p) for p in st[sids, 2] if p >= 0 and int(p) not in new_id]
        assert not bad, f"partners outside the increment: {bad[:5]}"
        ns[:, 2] = [new_id[int(p)] if p >= 0 else -1 for p in st[sids, 2]]
        out = {}
        for key, X in planes.items():
            Y = np.ascontiguousarray(X[:, idx])
            if key == "a_q" and a.lean:
                Y[:40] = 0
            out[key] = Y
        for key in FRAME:
            if key in pk.infos:
                out[key] = np.ascontiguousarray(np.asarray(pk.get(key))[idx])
        for key in STREAM:
            if key in pk.infos:
                out[key] = np.asarray(pk.get(key))[sids]
        for e in encs:                                   # an input that already carries encoder features (CANDOR)
            out[e] = np.ascontiguousarray(np.asarray(pk.get(e))[idx])
            for sfx in slim_pack.DERIVED + ("_lookahead",):
                if e + sfx in pk.infos:
                    out[e + sfx] = np.asarray(pk.get(e + sfx))
            if e + "_ok" in pk.infos:
                out[e + "_ok"] = np.asarray(pk.get(e + "_ok"))[sids]
        for key in globals_:
            out[key] = np.asarray(pk.get(key))
        hours = {}
        for r in ns:
            if r[5]:
                continue
            hk = f"{dsets[r[4]] if r[4] < len(dsets) else r[4]}/{['train', 'val', 'test'][r[3]]}"
            hours[hk] = hours.get(hk, 0) + r[1] / 30 / 3600
        info = dict(name=f"{a.name}_inc{f:g}", frac_lo=fracs[k - 1] if k else 0.0, frac_hi=f, seed=a.seed,
                    source=os.path.basename(a.pack), streams=int(len(sids)), frames=int(len(idx)),
                    sessions=int(len(np.unique(st[sids, 6]))), hours={kk: round(v, 3) for kk, v in hours.items()})
        out["streams"] = ns
        out["src_streams"] = sids.astype(np.int64)
        out["names"] = np.array(json.dumps(dict(names, hours=hours, increment=info)))
        out["pack_meta"] = np.array(json.dumps(dict(meta0, increment=info, lean_no_mel=bool(a.lean))))
        if a.lean:
            out["lean_no_mel"] = np.array(1)
        path = os.path.join(a.out_dir, f"{a.name}_inc{f:g}.npz")
        savez_atomic(path, out)
        del out
        log(f"increment {f:g}: {info['streams']} streams, {info['frames']} frames, {info['sessions']} sessions, "
            f"hours {info['hours']} -> {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
        rec = dict(info, path=path)
        if a.zstd:
            rec.update(zst(path, a.zstd))
        manifest["increments"].append(rec)
    manifest["built"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["seconds"] = round(time.time() - t0)
    mp = os.path.join(a.out_dir, f"{a.name}_sweep.json")
    old = json.load(open(mp)) if os.path.exists(mp) else {}
    manifest["enc"] = old.get("enc")
    with open(mp + ".tmp", "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(mp + ".tmp", mp)
    if a.ready:
        for rec in manifest["increments"]:
            ready(rec["name"], rec["path"], "sweep base increment" + (" (lean)" if a.lean else ""),
                  dict(zst_mb=rec.get("zst_mb"), frames=rec["frames"], streams=rec["streams"], hours=rec["hours"]))
    log(f"base done in {time.time() - t0:.0f}s -> {mp}")


def feature_files(rows, d):
    return [os.path.join(d, r["stream_id"] + ".npy") for r in rows]


def coverage(rows, d):
    have = set(os.listdir(d)) if os.path.isdir(d) else set()
    return sum(1 for r in rows if r["stream_id"] + ".npy" in have)


def cmd_enc(a):
    t0 = time.time()
    mp = os.path.join(a.out_dir, f"{a.name}_sweep.json")
    man = json.load(open(mp))
    src = Src(man["source"])
    st = np.asarray(src.get("streams"))
    N = int((st[:, 0] + st[:, 1]).max())
    rows = [json.loads(x) for x in open(a.streams) if x.strip()]
    d = os.path.join(a.feats_dir, a.enc)
    meta = json.load(open(os.path.join(d, "META.json"))) if os.path.exists(os.path.join(d, "META.json")) else {}
    by_stream = {}
    for r in rows:
        ps = int(r["pack_stream"])
        assert st[ps, 1] == r["n_frames"], (r["stream_id"], st[ps, 1], r["n_frames"])
        by_stream[ps] = os.path.join(d, r["stream_id"] + ".npy")
    present = {ps: f for ps, f in by_stream.items() if os.path.exists(f)}
    cov = len(present) / max(len(by_stream), 1)
    log(f"enc_{a.enc}: {len(present)}/{len(by_stream)} listed streams have features ({cov:.2%}); "
        f"{len(st) - len(by_stream)} pack streams are not in the list (no audio features -> ok 0)")
    if cov < a.min_coverage:
        raise SystemExit(f"coverage {cov:.2%} < --min-coverage {a.min_coverage:.0%}: not building")
    # ---- one quantization for every increment: per-dim mean + 99.99th pct of |x - mean| over TRAIN frames
    rng = np.random.default_rng(0)
    tr = [ps for ps in present if st[ps, 3] == 0]
    pick = rng.choice(tr, min(len(tr), 400), replace=False)
    X = np.concatenate([np.load(present[ps]).astype(np.float32) for ps in pick])
    X = X[np.isfinite(X).all(1)]
    D = X.shape[1]
    qmean = X.mean(0).astype(np.float32)
    scale = (np.maximum(np.quantile(np.abs(X - qmean), 0.9999, axis=0), 1e-3) / 127).astype(np.float32)
    if a.scale_from:                                 # same int8 grid as another pack (ConcatPack keeps int8 as-is)
        zs = np.load(a.scale_from)
        scale, qmean = np.asarray(zs[f"enc_{a.enc}_scale"], np.float32), np.asarray(zs[f"enc_{a.enc}_qmean"], np.float32)
        log(f"int8 scale/qmean from {a.scale_from}")
    clip_frac = float((np.abs(X - qmean) > 127 * scale).mean())
    del X
    log(f"quantization from {len(pick)} train streams: scale median {np.median(scale):.4f}, clipped {clip_frac:.5%}")
    # ---- int8 features in the SOURCE frame order (+ ok per source stream)
    E = np.zeros((N, D), np.int8)
    ok = np.zeros(len(st), np.uint8)
    bad = 0
    for ps, f in present.items():
        s0, n = int(st[ps, 0]), int(st[ps, 1])
        try:
            x = np.load(f).astype(np.float32)
        except Exception:                                  # noqa: BLE001  (unreadable: counts as missing)
            bad += 1
            continue
        if x.shape != (n, D):
            if x.ndim == 2 and x.shape[1] == D and abs(len(x) - n) <= 2:
                x = x[:n] if len(x) >= n else np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)])
            else:
                bad += 1
                continue
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        E[s0:s0 + n] = np.clip(np.round((x - qmean) / scale), -127, 127).astype(np.int8)
        ok[ps] = 1
    la = int(meta.get("lookahead_frames", 1))
    log(f"features loaded: ok {int(ok.sum())} streams ({bad} unreadable / bad shape), {time.time() - t0:.0f}s")
    key = f"enc_{a.enc}"
    common = {key + "_scale": scale, key + "_qmean": qmean, key + "_lookahead": np.array(la)}
    encrec = dict(encoder=a.enc, dims=D, lookahead_frames=la, streams_listed=len(by_stream), streams_ok=int(ok.sum()),
                  coverage_listed=round(int(ok.sum()) / max(len(by_stream), 1), 4), clip_frac=clip_frac,
                  meta=os.path.join(d, "META.json"), files=[])
    for rec in man["increments"]:
        zb = np.load(rec["path"], allow_pickle=False)
        sids = np.asarray(zb["src_streams"])
        idx = np.concatenate([np.arange(st[i, 0], st[i, 0] + st[i, 1]) for i in sids])
        out = dict(common)
        out[key] = np.ascontiguousarray(E[idx])
        out[key + "_ok"] = ok[sids]
        nb = np.asarray(zb["streams"])
        out["enc_for"] = np.array(json.dumps(dict(base=os.path.basename(rec["path"]), frames=int(len(idx)),
                                                  streams=int(len(sids)), frames_check=int((nb[:, 0] + nb[:, 1]).max()))))
        path = rec["path"][:-4] + ".enc.npz"
        savez_atomic(path, out)
        r2 = dict(path=path, inc=rec["name"], streams_ok=int(ok[sids].sum()), streams=int(len(sids)),
                  frames_ok=int(sum(st[i, 1] for i in sids if ok[i])), frames=int(len(idx)))
        log(f"{os.path.basename(path)}: {r2['streams_ok']}/{r2['streams']} streams with features "
            f"({r2['frames_ok'] / max(r2['frames'], 1):.1%} of frames), {os.path.getsize(path) / 1e6:.0f} MB")
        if a.zstd:
            r2.update(zst(path, a.zstd))
        encrec["files"].append(r2)
    if a.ext:
        out = dict(common)
        out[key] = E
        out[key + "_ok"] = ok
        savez_atomic(a.ext, out)
        encrec["ext"] = dict(path=a.ext, mb=round(os.path.getsize(a.ext) / 1e6), note="full_v4 = source + this as pack_extra")
        log(f"join {a.ext}: {os.path.getsize(a.ext) / 1e6:.0f} MB (full_v4 = {os.path.basename(man['source'])} + it)")
    man = json.load(open(mp))
    man["enc"] = encrec
    with open(mp + ".tmp", "w") as fh:
        json.dump(man, fh, indent=1)
    os.replace(mp + ".tmp", mp)
    if a.ready:
        for r2 in encrec["files"]:
            ready(os.path.basename(r2["path"])[:-4], r2["path"], f"sweep enc_{a.enc} int8 for {r2['inc']}",
                  dict(zst_mb=r2.get("zst_mb"), streams_ok=r2["streams_ok"], streams=r2["streams"]))
        if a.ext:
            ready("full_v4", a.ext, f"full_v4 = {os.path.basename(man['source'])} + this (enc_{a.enc} int8 + ok) as pack_extra")
    log(f"enc done in {time.time() - t0:.0f}s -> {mp}")


def pids_alive(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    for tok in open(path).read().split():
        if tok.isdigit() and os.path.exists(f"/proc/{tok}"):
            out.append(int(tok))
    return out


def cmd_watch(a):
    rows = [json.loads(x) for x in open(a.streams) if x.strip()]
    d = os.path.join(a.feats_dir, a.enc)
    last, hist, t_log = None, [], 0.0
    tot = sum(r["n_frames"] for r in rows)
    while True:
        have = set(os.listdir(d)) if os.path.isdir(d) else set()
        done = [r for r in rows if r["stream_id"] + ".npy" in have]
        n, fr = len(done), sum(r["n_frames"] for r in done)
        alive = pids_alive(a.pids_file)
        hist.append((time.time(), fr))
        hist = [h for h in hist if h[0] > time.time() - 1800]
        rate = (hist[-1][1] - hist[0][1]) / max(hist[-1][0] - hist[0][0], 1) if len(hist) > 1 else 0.0
        eta = (time.strftime("%H:%M", time.localtime(time.time() + (tot - fr) / rate)) if rate > 0 else "?")
        msg = f"watch: {n}/{len(rows)} feature files, extraction pids alive {len(alive)}"
        if msg != last and time.time() - t_log > 600 or n >= len(rows):
            log(f"{msg}; frames {fr}/{tot} ({fr / max(tot, 1):.1%}), {rate * 60:.0f} frames/min over the last "
                f"{(hist[-1][0] - hist[0][0]) / 60:.0f} min -> ETA {eta}")
            last, t_log = msg, time.time()
        if n >= len(rows) or (not alive and n / max(len(rows), 1) >= a.min_coverage):
            time.sleep(5 if n >= len(rows) else 0)         # let a final os.replace settle
            log("watch: starting enc build")
            return cmd_enc(a)
        if not alive and a.pids_file and os.path.exists(a.pids_file):
            log(f"watch: extraction finished with coverage {n / max(len(rows), 1):.2%} < {a.min_coverage:.0%} - "
                f"NOT building (rerun `enc` with a lower --min-coverage if that is acceptable)")
            return 2
        time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("base")
    b.add_argument("--pack", required=True)
    for p in (b,) + tuple(sub.add_parser(c) for c in ("enc", "watch")):
        p.add_argument("--out-dir", default=os.path.join(P, "packs", "sweep"))
        p.add_argument("--name", default="fv4")
        p.add_argument("--zstd", type=int, default=12, help="zstd level for the .zst copies (0 = none)")
        p.add_argument("--no-ready", dest="ready", action="store_false", help="do not append READY lines")
        if p is not b:
            p.add_argument("--streams", default=os.path.join(P, "streams_full.jsonl"))
            p.add_argument("--enc", default="whisper_tiny")
            p.add_argument("--feats-dir", default=os.path.join(P, "feats"))
            p.add_argument("--min-coverage", type=float, default=0.98)
            p.add_argument("--ext", default=None, help="also write the join (whole pack, source frame order)")
            p.add_argument("--pids-file", default=os.path.join(P, "audio_work", "pids_wt_full.txt"))
            p.add_argument("--poll", type=float, default=60)
            p.add_argument("--scale-from", default=None, help="an existing .enc.npz whose int8 scale/qmean to reuse")
    b.add_argument("--fracs", default="25,50,100")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-lean", dest="lean", action="store_false")
    b.add_argument("--train-only", action="store_true", help="drop val/test streams (an extra corpus, e.g. CANDOR)")
    a = ap.parse_args()
    return {"base": cmd_base, "enc": cmd_enc, "watch": cmd_watch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
