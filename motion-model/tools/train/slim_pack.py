"""v3 pack -> SLIM training pack (smaller to store and to move between machines).

    python -I tools/train/slim_pack.py <v3.qpack.npz | v3.qpack.npz.zst> <out.npz>
           [--encoders whisper_tiny,hubert_base] [--pca-dims 32] [--no-a2f] [--fit-frames 400000] [--seed 0] [--zstd]

Members of the output:
  * every BASE member unchanged, byte for byte (qpack planes m_lo/m_hi/a_q/mask/wexp, streams, quality, names,
    mean/std/amean/astd, fl_*/fr_*, style/style_tables/pack_meta, anything else that is not one of the below);
  * `events` u8 (N,3) as-is;
  * `a2f_u8` u8 (N,52) = round(clip(a2f, 0, 1) * 255) (a2f / 255 back) and `a2f_mask` u8 (N,); --no-a2f drops
    a2f, a2f_u8 and a2f_mask;
  * per selected encoder (--encoders, default every `enc_<name>` present; the rest are dropped), with k = --pca-dims:
      enc_<name>          int8 (N,k)  PCA coefficients, q = clip(round(c / scale), -127, 127)
      enc_<name>_scale    f32 (k,)    4 * std(c) / 127 over the fit frames
      enc_<name>_qmean    f32 (k,)    0 (centred PCA); dequant c = q * scale + qmean
      enc_<name>_pca_mean f32 (D,)    proj(f) = (f - pca_mean) @ pca_V.T
      enc_<name>_pca_V    f32 (k,D)
      enc_<name>_pca_ev   f32 ()      explained variance fraction of the k components (fit frames)
    PCA is fitted on <= --fit-frames random TRAIN non-ghost frames (streams split==0, ghost==0), two passes in f64
    (mean, then the centred covariance). --pca-dims 0 keeps the encoder as float16 (N,D) unchanged.
BLAS threads: SLIM_THREADS (default 8). Large members are streamed: uncompressed .npz members are memory-mapped from the zip, projections run in row
blocks (never a float32 (N,D) copy). A .zst input is decompressed next to the output and removed afterwards.
--zstd also writes <out>.zst (system zstd -T8 -10).
"""
import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile

# shared hosts: cap BLAS threads (unbounded OpenBLAS spun 88 threads for 2 min on the synthetic test)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SLIM_THREADS", "8"))
import numpy as np  # noqa: E402

BLOCK = 65536
DERIVED = ("_scale", "_qmean", "_pca_mean", "_pca_V", "_pca_ev")


# ---------------------------------------------------------------------------------------- npz plumbing
def _member_infos(path):
    with zipfile.ZipFile(path) as zf:
        return {i.filename[:-4] if i.filename.endswith(".npy") else i.filename: i for i in zf.infolist()}


def open_member(path, info):
    """np.memmap of a STORED (uncompressed) .npy member of a zip, or None when it cannot be mapped."""
    if info.compress_type != zipfile.ZIP_STORED:
        return None
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        if h[:4] != b"PK\x03\x04":
            return None
        n, m = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        off = f.tell()
    if dtype.hasobject:
        return None
    if int(np.prod(shape)) == 0:
        return np.zeros(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, order="F" if fortran else "C", offset=off)


class Pack:
    """Lazy access to the members of an .npz: memmap when stored, else a full load (np.load)."""

    def __init__(self, path):
        self.path = path
        self.infos = _member_infos(path)
        self._z = None

    @property
    def files(self):
        return list(self.infos)

    def __getitem__(self, k):
        mm = open_member(self.path, self.infos[k])
        if mm is not None:
            return mm
        if self._z is None:
            self._z = np.load(self.path, allow_pickle=False)
        print(f"  note: {k} is compressed in the zip -> loaded whole", flush=True)
        return self._z[k]

    def copy_raw(self, k, zout):
        """stream the member's .npy bytes unchanged into the output zip."""
        info = self.infos[k]
        with zipfile.ZipFile(self.path) as zin, zin.open(info) as src, \
                zout.open(info.filename, "w", force_zip64=True) as dst:
            shutil.copyfileobj(src, dst, 1 << 24)
        return info.file_size


def write_member(zout, name, arr):
    arr = np.asanyarray(arr)
    with zout.open(name + ".npy", "w", force_zip64=True) as f:
        np.lib.format.write_array(f, arr, allow_pickle=False)
    return arr.nbytes + 128


# ---------------------------------------------------------------------------------------- PCA
def train_rows(streams, n_max, seed):
    st = np.asarray(streams)
    tr = [np.arange(s0, s0 + n) for s0, n, _, sp, _, gh, _ in st if sp == 0 and gh == 0]
    tr = np.concatenate(tr) if tr else np.zeros(0, np.int64)
    if len(tr) > n_max:
        tr = np.random.default_rng(seed).choice(tr, n_max, replace=False)
    return np.sort(tr)


def fit_pca(X, rows, k, block=BLOCK):
    """X (N,D) any float dtype (memmap ok), rows: fit rows -> (mean f64 (D,), V f64 (k,D), ev, coef std (k,))."""
    D = X.shape[1]
    s, cnt = np.zeros(D), 0
    for r0 in range(0, len(rows), block):
        xb = np.asarray(X[rows[r0:r0 + block]], np.float64)
        xb = xb[np.isfinite(xb).all(1)]
        s += xb.sum(0); cnt += len(xb)
    if cnt < 2:
        raise SystemExit("slim_pack: fewer than 2 finite TRAIN non-ghost frames to fit the PCA on")
    mu = s / cnt
    C = np.zeros((D, D))
    for r0 in range(0, len(rows), block):
        xb = np.asarray(X[rows[r0:r0 + block]], np.float64)
        xb = xb[np.isfinite(xb).all(1)] - mu
        C += xb.T @ xb
    C /= cnt - 1
    w, U = np.linalg.eigh(C)                              # ascending
    w, U = w[::-1], U[:, ::-1]
    w = np.maximum(w, 0)
    k = min(k, D)
    V = U[:, :k].T.copy()
    # deterministic sign: the largest-|.| loading of each component positive
    sg = np.sign(V[np.arange(k), np.abs(V).argmax(1)]); sg[sg == 0] = 1
    V *= sg[:, None]
    ev = float(w[:k].sum() / max(w.sum(), 1e-30))
    return mu, V, ev, np.sqrt(w[:k])


def project_q8(X, mu, V, scale, block=BLOCK):
    """X (N,D) -> int8 (N,k) = clip(round(((X - mu) @ V.T) / scale)); non-finite rows -> 0 (the mean)."""
    N = X.shape[0]
    k = V.shape[0]
    q = np.empty((N, k), np.int8)
    mu32, Vt32, inv = mu.astype(np.float32), np.ascontiguousarray(V.T, np.float32), (1.0 / scale).astype(np.float32)
    clipped = 0
    for r0 in range(0, N, block):
        xb = np.asarray(X[r0:r0 + block], np.float32)
        bad = ~np.isfinite(xb).all(1)
        if bad.any():
            xb = np.where(np.isfinite(xb), xb, mu32)
        c = ((xb - mu32) @ Vt32) * inv
        c[bad] = 0
        r = np.rint(c)
        clipped += int((np.abs(r) > 127).sum())
        q[r0:r0 + block] = np.clip(r, -127, 127).astype(np.int8)
    return q, clipped / max(N * k, 1)


def to_u8_unit(X, block=BLOCK):
    out = np.empty(X.shape, np.uint8)
    for r0 in range(0, X.shape[0], block):
        xb = np.nan_to_num(np.asarray(X[r0:r0 + block], np.float32))
        out[r0:r0 + block] = np.rint(np.clip(xb, 0, 1) * 255).astype(np.uint8)
    return out


def to_f16(X, block=BLOCK):
    out = np.empty(X.shape, np.float16)
    for r0 in range(0, X.shape[0], block):
        out[r0:r0 + block] = np.asarray(X[r0:r0 + block], np.float32).astype(np.float16)
    return out


# ---------------------------------------------------------------------------------------- main
def encoders_in(files):
    return sorted(k[4:] for k in files if k.startswith("enc_") and not k.endswith(DERIVED))


def slim(inp, out, encoders=None, pca_dims=32, no_a2f=False, fit_frames=400_000, seed=0, log=print):
    t0 = time.time()
    P = Pack(inp)
    files = P.files
    present = encoders_in(files)
    sel = present if encoders is None else list(encoders)
    for e in sel:
        if e not in present:
            raise SystemExit(f"slim_pack: encoder {e!r} not in the pack (present: {present})")
    if any(k.endswith(DERIVED) and k.startswith("enc_") for k in files):
        log("warning: the input already has enc_*_scale/_pca_* members (a slim pack?): they are copied as base")
    special = {"a2f", "a2f_mask", "events"} | {f"enc_{e}" for e in present}
    sizes, info = {}, {}
    tmp = out + ".tmp.npz"
    streams = np.asarray(P["streams"])
    N = int(streams[:, 0].max() + streams[streams[:, 0].argmax(), 1]) if len(streams) else 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
        for k in files:
            if k in special:
                continue
            sizes[k] = P.copy_raw(k, zout)
        if "events" in files:
            ev = np.asarray(P["events"])
            if ev.dtype != np.uint8:
                ev = (np.nan_to_num(ev.astype(np.float32)) != 0).astype(np.uint8) if ev.dtype.kind == "f" \
                    else np.clip(ev, 0, 255).astype(np.uint8)
            sizes["events"] = write_member(zout, "events", ev)
            del ev
        if "a2f" in files and not no_a2f:
            A = P["a2f"]
            assert A.shape[0] == N, (A.shape, N)
            sizes["a2f_u8"] = write_member(zout, "a2f_u8", to_u8_unit(A))
            del A
        if "a2f_mask" in files and not no_a2f:
            sizes["a2f_mask"] = write_member(zout, "a2f_mask", (np.asarray(P["a2f_mask"]) != 0).astype(np.uint8))
        rows = None
        for e in sel:
            X = P[f"enc_{e}"]
            assert X.ndim == 2 and X.shape[0] == N, (e, X.shape, N)
            D = X.shape[1]
            if pca_dims <= 0:
                sizes[f"enc_{e}"] = write_member(zout, f"enc_{e}", X if X.dtype == np.float16 else to_f16(X))
                info[e] = dict(D=D, kept="float16")
                log(f"enc_{e}: kept float16 ({N} x {D})")
                continue
            if rows is None:
                rows = train_rows(streams, fit_frames, seed)
            tf = time.time()
            mu, V, evf, csd = fit_pca(X, rows, pca_dims)
            k = V.shape[0]
            scale = (4.0 * np.maximum(csd, 1e-8) / 127.0).astype(np.float32)
            q, clip = project_q8(X, mu, V, scale)
            for nm, arr in ((f"enc_{e}", q), (f"enc_{e}_scale", scale), (f"enc_{e}_qmean", np.zeros(k, np.float32)),
                            (f"enc_{e}_pca_mean", mu.astype(np.float32)), (f"enc_{e}_pca_V", V.astype(np.float32)),
                            (f"enc_{e}_pca_ev", np.float32(evf))):
                sizes[nm] = write_member(zout, nm, arr)
            info[e] = dict(D=D, k=k, explained_variance=round(evf, 4), clipped_frac=clip, fit_frames=int(len(rows)),
                           seconds=round(time.time() - tf, 1))
            log(f"enc_{e}: PCA {D} -> {k} explains {evf:.4f} of the variance (fit on {len(rows)} train frames), "
                f"int8 clipped {clip * 100:.3f}%")
            del X, q
    os.replace(tmp, out)
    total = os.path.getsize(out)
    sizes = {k: i.file_size for k, i in _member_infos(out).items()}       # the real .npy bytes in the output
    for k in sorted(sizes, key=lambda k: -sizes[k]):
        log(f"  {k:28s} {sizes[k] / 1e6:10.2f} MB")
    log(f"total {total / 1e6:.2f} MB ({out}); input {os.path.getsize(inp) / 1e6:.2f} MB; {time.time() - t0:.1f}s")
    return dict(out=out, bytes=total, members={k: int(v) for k, v in sizes.items()}, encoders=info, frames=N)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--encoders", default=None, help="comma list (default: every enc_* in the pack)")
    ap.add_argument("--pca-dims", type=int, default=32, help="0 = keep the encoders as float16")
    ap.add_argument("--no-a2f", action="store_true", help="drop a2f / a2f_mask")
    ap.add_argument("--fit-frames", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zstd", action="store_true", help="also write <out>.zst (zstd -T8 -10)")
    a = ap.parse_args(argv)
    if a.zstd and not shutil.which("zstd"):
        ap.error("--zstd: no zstd binary on PATH")
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    inp, dec = a.inp, None
    if inp.endswith(".zst"):
        if not shutil.which("zstd"):
            ap.error("a .zst input needs the zstd binary")
        dec = out + ".in.tmp.npz"
        print(f"decompressing {inp} -> {dec}", flush=True)
        subprocess.run(["zstd", "-d", "-q", "-f", "-T8", "--long=31", inp, "-o", dec], check=True)
        inp = dec
    try:
        enc = [e for e in a.encoders.split(",") if e] if a.encoders else None
        res = slim(inp, out, encoders=enc, pca_dims=a.pca_dims, no_a2f=a.no_a2f, fit_frames=a.fit_frames,
                   seed=a.seed, log=lambda m: print(m, flush=True))
    finally:
        if dec and os.path.exists(dec):
            os.remove(dec)
    if a.zstd:
        subprocess.run(["zstd", "-q", "-f", "-T8", "-10", out, "-o", out + ".zst"], check=True)
        res["zst_bytes"] = os.path.getsize(out + ".zst")
        print(f"zst {res['zst_bytes'] / 1e6:.2f} MB ({out}.zst)", flush=True)
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
