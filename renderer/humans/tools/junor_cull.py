#!/usr/bin/env python3
"""BODY: cull the layers that can never be seen (the game trick): blouse under the cardigan, body skin under any garment.

A vertex of an INNER layer is 'covered' when a ray along its outward normal hits an OUTER layer within --reach metres in the
rest pose.  The covered set is eroded by --erode vertex rings (keeps a margin under every opening: V neck, cuffs, hem,
collar), and a triangle whose three vertices stay covered is made degenerate (all three indices = its first one) — index
counts, submesh ranges, shapes and every buffer stay as they are, the GPU just rasterises nothing there.  Skinned layers
that deform differently (blouse vs cardigan at a raised shoulder) then have nothing left underneath to poke through.

  python renderer/humans/tools/junor_cull.py ASSET_DIR [--reach 0.08] [--erode 2] [--dry]
Idempotent (re-running on a culled asset finds the same set; degenerate triangles are skipped).  Writes
juno.asset.json meta "culled": {lod: {submesh: n_tris}}.  Run after every HEAD asset build (+ HAIR regroom), like --reach.
"""
import argparse, json, os, sys
import numpy as np

RINGS = 4          # the weight transfer fades in over this many vertex rings inside the covered region (visible parts keep theirs)
INNER_OUTER = [   # (inner submesh, [outer submeshes])
    ("cloth_blouse", ["cloth_cardigan", "cloth_cardigan_rib", "cloth_buttons"]),
    ("body", ["cloth_blouse", "cloth_cardigan", "cloth_cardigan_rib", "cloth_trousers", "shoes", "shoes_sole"]),
    ("cloth_trousers", ["shoes", "shoes_sole"]),
    ("head", ["cloth_blouse", "cloth_cardigan", "cloth_cardigan_rib"]),        # the scan head's shoulder / chest skirt
    ("neck_bridge", ["cloth_blouse", "cloth_cardigan", "cloth_cardigan_rib"]),
]


def load(lod_dir):
    m = json.load(open(os.path.join(lod_dir, "mesh.json")))
    raw = open(os.path.join(lod_dir, "mesh.bin"), "rb").read()
    B = m["buffers"]; N = m["vertex_count"]; M = m["index_count"]
    def arr(n, cnt):
        b = B[n]
        return np.frombuffer(raw, dtype=b["dtype"], count=cnt*b["comps"], offset=b["offset"]).reshape(cnt, b["comps"])
    return m, raw, arr("position", N).astype(np.float64), arr("normal", N).astype(np.float64), arr("index", M)[:, 0].copy()


def ray_hits(O, D, tris, reach):
    """Moller-Trumbore, any hit with 1e-4 < t < reach. O, D (n,3); tris (m,3,3) -> bool (n,)"""
    hit = np.zeros(len(O), bool)
    v0 = tris[:, 0]; e1 = tris[:, 1] - v0; e2 = tris[:, 2] - v0
    for a in range(0, len(O), 256):
        o = O[a:a+256, None, :]; d = D[a:a+256, None, :]
        p = np.cross(d, e2[None]); det = np.einsum("nmk,mk->nm", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0/np.where(ok, det, 1.0), 0.0)
        s = o - v0[None]
        u = np.einsum("nmk,nmk->nm", s, p)*inv
        q = np.cross(s, e1[None])
        v = np.einsum("nmk,nmk->nm", np.broadcast_to(d, q.shape), q)*inv
        t = np.einsum("nmk,mk->nm", q, e2)*inv
        h = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < reach)
        hit[a:a+256] = h.any(1)
    return hit


def cull_lod(lod_dir, reach, erode, dry):
    m, raw, P, Nn, I = load(lod_dir)
    sm = {s["name"]: s for s in m["submeshes"]}
    glob = int(I[sm["body"]["first_index"]:sm["body"]["first_index"] + sm["body"]["index_count"]].min()) >= sm["body"]["first_vertex"]
    def tris_of(s):
        t = I[s["first_index"]:s["first_index"] + s["index_count"]].reshape(-1, 3).astype(np.int64)
        return t if glob else t + s["first_vertex"]
    out = {}; depth = np.zeros(len(P))
    for inner, outers in INNER_OUTER:
        if inner not in sm:
            continue
        s = sm[inner]
        T = tris_of(s)
        live = ~((T[:, 0] == T[:, 1]) & (T[:, 1] == T[:, 2]))
        OT = np.concatenate([tris_of(sm[o]) for o in outers if o in sm])
        OT = OT[~((OT[:, 0] == OT[:, 1]) & (OT[:, 1] == OT[:, 2]))]
        vs = np.arange(s["first_vertex"], s["first_vertex"] + s["vertex_count"])
        n = Nn[vs]; n = n/np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        lo, hi = P[OT.ravel()].min(0) - reach, P[OT.ravel()].max(0) + reach    # only vertices near the outer layers
        near = np.all((P[vs] >= lo) & (P[vs] <= hi), axis=1)
        cov = np.zeros(len(vs), bool)
        cov[near] = ray_hits(P[vs][near], n[near], P[OT], reach)
        c = np.zeros(len(P), bool); c[vs] = cov
        lt = T[live]
        c0 = None
        for r in range(max(erode, RINGS)):      # a vertex stays covered only if all its triangle neighbours are
            if r == erode:
                c0 = c.copy()
            depth[c] += 1.0
            bad = ~c[lt].all(1)
            unc = np.zeros(len(P), bool); unc[lt[bad].ravel()] = True
            c &= ~unc
        depth[c] += 1.0
        c = c if c0 is None else c0
        kill = live & c[T].all(1)
        out[inner] = int(kill.sum())
        if not dry and kill.any():
            seg = I[s["first_index"]:s["first_index"] + s["index_count"]].reshape(-1, 3)
            seg[kill] = seg[kill][:, :1]
            I[s["first_index"]:s["first_index"] + s["index_count"]] = seg.ravel()
        print(f"  {os.path.basename(lod_dir)} {inner}: {int(cov.sum())}/{len(vs)} verts covered, {int(kill.sum())}/{int(live.sum())} tris culled", flush=True)
    if not dry:
        b = m["buffers"]["index"]
        ba = bytearray(raw); ba[b["offset"]:b["offset"] + I.nbytes] = I.astype(np.uint32).tobytes()
        open(os.path.join(lod_dir, "mesh.bin"), "wb").write(bytes(ba))
    return out, np.clip(depth/float(RINGS + 1), 0.0, 1.0)


BIND = [   # (layer, [layers whose skinning it adopts near them], gate)
    # hidden: only where the layer itself is covered (coverage depth g, fading in over RINGS rings from every opening):
    # the visible V / collar / neck keep their own weights.  src_visible: the cardigan near a VISIBLE blouse part adopts
    # the blouse's (original) weights instead — at the collar base / yoke the two layers then share one skinning too.
    ("cloth_blouse", ["cloth_cardigan", "cloth_cardigan_rib"], "hidden"),
    ("cloth_cardigan", ["cloth_blouse"], "src_visible"),
    ("cloth_cardigan_rib", ["cloth_blouse"], "src_visible"),
    ("body", ["cloth_blouse", "cloth_cardigan", "cloth_cardigan_rib"], "hidden"),
    ("head", ["cloth_blouse", "cloth_cardigan", "cloth_cardigan_rib"], "hidden"),     # chest / shoulder skin of the scan head
    ("neck_bridge", ["cloth_blouse", "cloth_cardigan", "cloth_cardigan_rib"], "hidden"),
]
HAND = ("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_", "neck_", "head")


def bind_lod(lod_dir, bones, d0, d1, dry, g=None):
    """Inner layers adopt the skin weights of the nearest outer-layer vertex (full within d0, fading out by d1, metres):
    layers that share one skinning keep their rest offsets in every pose, so a raised shoulder no longer pushes the
    blouse / skin out through the cardigan.  Vertices led by a hand / finger / neck / head bone keep their own weights."""
    m = json.load(open(os.path.join(lod_dir, "mesh.json")))
    raw = bytearray(open(os.path.join(lod_dir, "mesh.bin"), "rb").read())
    B = m["buffers"]; N = m["vertex_count"]
    P = np.frombuffer(bytes(raw), "f4", N*3, B["position"]["offset"]).reshape(N, 3).astype(np.float64)
    Jt = np.frombuffer(bytes(raw), "u2", N*4, B["joints"]["offset"]).reshape(N, 4).copy()
    W = np.frombuffer(bytes(raw), "f4", N*4, B["weights"]["offset"]).reshape(N, 4).copy()
    sm = {x["name"]: x for x in m["submeshes"]}
    keep_bone = np.array([b.startswith(HAND) for b in bones])
    out = {}
    J0, W0 = Jt.copy(), W.copy()             # src_visible reads the blouse's ORIGINAL weights
    for inner, outers, gate in BIND:
        if inner not in sm:
            continue
        vs = np.arange(sm[inner]["first_vertex"], sm[inner]["first_vertex"] + sm[inner]["vertex_count"])
        ov = np.concatenate([np.arange(sm[o]["first_vertex"], sm[o]["first_vertex"] + sm[o]["vertex_count"]) for o in outers if o in sm])
        near = np.empty(len(vs), np.int64); dist = np.empty(len(vs))
        for a in range(0, len(vs), 512):
            d = np.linalg.norm(P[vs[a:a+512], None, :] - P[None, ov, :], axis=2)
            k = d.argmin(1); near[a:a+512] = ov[k]; dist[a:a+512] = d[np.arange(len(k)), k]
        lead = Jt[vs, np.argmax(W[vs], axis=1)]
        f = np.clip((d1 - dist)/(d1 - d0), 0.0, 1.0)*(~keep_bone[lead])
        if g is not None and gate == "hidden":
            f = f*g[vs]
        elif g is not None and gate == "src_visible":
            f = f*(1.0 - g[near])
        SJ, SW = (J0, W0) if gate == "src_visible" else (Jt, W)
        n = 0
        for i, v in enumerate(vs):
            if f[i] <= 0.0:
                continue
            acc = {}
            for j, w in zip(Jt[v], W[v]):
                if w > 0: acc[int(j)] = acc.get(int(j), 0.0) + (1.0 - f[i])*float(w)
            for j, w in zip(SJ[near[i]], SW[near[i]]):
                if w > 0: acc[int(j)] = acc.get(int(j), 0.0) + f[i]*float(w)
            top = sorted(acc.items(), key=lambda x: -x[1])[:4]
            tw = sum(w for _, w in top)
            Jt[v] = [j for j, _ in top] + [0]*(4 - len(top))
            W[v] = [w/tw for _, w in top] + [0.0]*(4 - len(top))
            n += 1
        out[inner] = n
        print(f"  {os.path.basename(lod_dir)} {inner}: {n}/{len(vs)} verts rebound to {'+'.join(outers)}", flush=True)
    if not dry:
        raw[B["joints"]["offset"]:B["joints"]["offset"] + Jt.nbytes] = Jt.astype(np.uint16).tobytes()
        raw[B["weights"]["offset"]:B["weights"]["offset"] + W.nbytes] = W.astype(np.float32).tobytes()
        open(os.path.join(lod_dir, "mesh.bin"), "wb").write(bytes(raw))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asset"); ap.add_argument("--reach", type=float, default=0.08); ap.add_argument("--erode", type=int, default=3)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--bind", type=float, nargs=2, default=[0.03, 0.06], help="weight transfer full within / gone by (m); 0 0 = off")
    ap.add_argument("--no-cull", action="store_true")
    a = ap.parse_args()
    mp = os.path.join(a.asset, "juno.asset.json")
    meta = json.load(open(mp))
    res = {}
    for L in meta.get("lods", [{"name": "lod0"}]):
        d = os.path.join(a.asset, L.get("name", "lod0"))
        if os.path.exists(os.path.join(d, "mesh.json")):
            if a.bind[1] > 0 and "culled" not in meta:     # (once: a re-run would blend the weights again)
                sk = json.load(open(os.path.join(a.asset, meta.get("skeleton", "skeleton.json"))))
                g = cull_lod(d, a.reach, a.erode, True)[1]          # coverage depth (rings inside the covered region)
                bind_lod(d, [b["name"] for b in sk["bones"]], a.bind[0], a.bind[1], a.dry, g)
            if not a.no_cull:
                res[L.get("name")] = cull_lod(d, a.reach, a.erode, a.dry)[0]
    if not a.dry:
        meta["culled"] = {"by": "renderer/humans/tools/junor_cull.py", "reach": a.reach, "erode": a.erode, "tris": res,
                          "bind_m": a.bind}
        json.dump(meta, open(mp, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
