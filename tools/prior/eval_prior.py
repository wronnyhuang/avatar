"""Free-running generation on held-out videos (real live-available inputs, sampled motion) -> the same
statistics as the real people of those videos, side by side; plus a 10-minute stability run.
Usage: python -I eval_prior.py <scratch> <models_dir> <tag> [--temp 1.0] [--subset all|dyad] [--out json]"""
import json, sys, os, argparse
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from prior_model import Prior, OUT
from motionlib import COLS
import mstats

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("mdir"); ap.add_argument("tag")
ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--subset", default="all")
ap.add_argument("--out", default=None); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--H", type=int, default=128); ap.add_argument("--K", type=int, default=4)
ap.add_argument("--maxlen", type=int, default=3000); ap.add_argument("--chunk", type=int, default=1)
ap.add_argument("--chtemp", default="", help="per-group temperature, e.g. head=1.4,gaze=0.9,brow=1.2,smile=1,eye=1,arm=1")
ap.add_argument("--leak", default="", help="per-group leak toward neutral per frame, e.g. gaze=0.02,smile=0.005")
ap.add_argument("--smooth", type=float, default=0.0, help="output one-pole low-pass (Hz) on head angles+translation")
ap.add_argument("--speak-head", type=float, default=0.0, help="head temperature while SELF speaking (0 = same as head)")
ap.add_argument("--anchor", type=float, default=0.0, help="per-frame pull of WORLD gaze (eye+head) toward the partner")
ap.add_argument("--people", default="val", help="val | all (free-run every person; compare with the full population)")
ap.add_argument("--nod", type=int, default=0); ap.add_argument("--nodgain", type=float, default=1.0)
ap.add_argument("--subset-people", default="", help="dyad: only people with a visible partner")
ap.add_argument("--zero-pf", type=int, default=0, help="ablation: hide the partner's face from the model (inputs 8..16 = 0)")
ap.add_argument("--smile-hz", type=float, default=0.0, help="output one-pole low-pass on smile (Hz)")
ap.add_argument("--idle-head", type=float, default=0.0, help="head temperature when nobody speaks (0 = same as head)")
ap.add_argument("--blink-saccade", type=float, default=1.0, help="blink hazard multiplier in a chunk with a >= 8 deg gaze shift")
ap.add_argument("--rx", default="", help="reaction model tag: modulate smile / nod hazard / head energy / gaze anchor")
ap.add_argument("--react", default="smile=1,nod=1,gaze=1,energy=0.5")
a = ap.parse_args()
torch.set_num_threads(4); torch.manual_seed(a.seed); np.random.seed(a.seed)
root = Path(a.root); md = Path(a.mdir)
D = dict(np.load(root / "dataset.npz")); meta = json.loads((root / "dataset_meta.json").read_text())
val_ids = set(json.loads((root / "split.json").read_text())["val"])
nz = np.load(md / f"{a.tag}_norm.npz"); pos_c, pos_s, d_s = nz["pos_c"], nz["pos_s"], nz["d_s"]
CH = a.chunk
GROUPS = dict(head=[0, 1, 2, 3, 4, 5], ang=[0, 1, 2], trans=[3, 4, 5], gaze=[6, 7], brow=[8, 9], smile=[10], eye=[11, 12], arm=[13])
def per_group(spec, default):
    v = np.full(len(OUT), default, np.float32)
    for kv in filter(None, spec.split(",")):
        k, x = kv.split("="); v[GROUPS[k]] = float(x)
    return v
TEMPV = torch.from_numpy(per_group(a.chtemp, a.temp))
LEAKV = torch.from_numpy(per_group(a.leak, 0.0))
model = Prior(H=a.H, K=a.K, C=CH, nod=bool(a.nod)); model.load_state_dict(torch.load(md / f"{a.tag}.pt")); model.eval()
cols = [COLS.index(c) for c in OUT]
ext_all = np.nan_to_num(np.concatenate([D["sv"], D["pv"], D["pf"]], 1).astype(np.float32))


def nod_template(k, A, double):
    """pitch offset (deg, + = nose down) k frames into a nod: a 0.45 s sine dip, optional 0.6x second dip"""
    T1 = 14
    if k < T1:
        return A * np.sin(np.pi * k / T1)
    if double and k < 2 * T1:
        return 0.6 * A * np.sin(np.pi * (k - T1) / T1)
    return 0.0


RXM = None
if a.rx:
    from prior_model import RX
    _sd = torch.load(md / f"{a.rx}.pt"); RXM = RX(H=_sd["gru.weight_hh_l0"].shape[1]); RXM.load_state_dict(_sd); RXM.eval()
    RXW = json.loads((md / f"{a.rx}_weights.json").read_text())
    RXN = RXW["norm"]; RXB = RXW["meta"]["eval"]["base_rates"]
    REACT = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.react.split(",")}


def rx_outputs(ext):
    x = np.concatenate([ext[:, :, 8:17], ext[:, :, 4:8], ext[:, :, 3:4]], 2)
    with torch.no_grad():
        o, _ = RXM(torch.from_numpy(x))
    o = o.numpy(); sg = lambda v: 1 / (1 + np.exp(-v))
    return dict(smile_rel=o[..., 0] * RXN["ysd"][0] + RXN["ymu"][0], head_speed=np.expm1(o[..., 1] * RXN["ysd"][1] + RXN["ymu"][1]),
                bignod=sg(o[..., 4]), avert=sg(o[..., 5]), present=ext[:, :, 8] > 0)


def generate(ext, y0, temp, seed=0):
    """ext (B,T,17) inputs; y0 (B,D) normalized start -> Y (B,T,D) normalized, blinks (B,T).
    Frame 0 = y0; each step at anchor a sees mean(ext[a-CH+1..a]) and y[a], emits frames a+1..a+CH."""
    g = torch.Generator().manual_seed(seed)
    B, T, _ = ext.shape; Dd = len(OUT)
    Y = np.zeros((B, T, Dd), np.float32); BL = np.zeros((B, T), np.float32)
    y = torch.from_numpy(y0.astype(np.float32)); h = None
    Y[:, 0] = y.numpy()
    global NODT, NODA, NOD2, NODOFF, rng_np
    rng_np = np.random.default_rng(seed)
    NODT = np.full(B, -1); NODA = np.zeros(B); NOD2 = np.zeros(B, bool); NODOFF = np.zeros((B, T), np.float32)
    clock = torch.ones(B); armvis = torch.ones(B, 1)
    E = torch.from_numpy(ext)
    dS = torch.from_numpy(d_s.astype(np.float32))
    ar = torch.arange(B)
    RO = rx_outputs(ext) if RXM is not None else None
    with torch.no_grad():
        for an in range(0, T - 1, CH):
            e = E[:, max(0, an - CH + 1):an + 1].mean(1)
            x = torch.cat([e, y, armvis, clock[:, None]], 1)[:, None]
            hd, h = model(x, h)
            lg, mu, ls, b = hd[0][:, 0], hd[1][:, 0], hd[2][:, 0], hd[3][:, 0]
            if a.nod:
                pn = torch.sigmoid(hd[4][:, 0]) * a.nodgain
                if RO is not None:
                    mul = np.clip(RO["bignod"][:, an] / max(RXB["bignod"], 1e-3), 0.3, 3) ** REACT["nod"]
                    pn = pn * torch.from_numpy(np.where(RO["present"][:, an], mul, 1.0).astype(np.float32))
                fire = (torch.rand(B, generator=g) < pn).numpy()
                for bi in np.where(fire)[0]:
                    if NODT[bi] < 0 or NODT[bi] > 18:
                        NODT[bi] = 0; NODA[bi] = float(np.clip(np.exp(np.log(6.0) + 0.3 * rng_np.standard_normal()), 4, 12))
                        NOD2[bi] = rng_np.random() < 0.3
            k = torch.multinomial(torch.softmax(lg, -1), 1, generator=g)[:, 0]
            m = mu[ar, k]; s = ls[ar, k].exp()
            tv = TEMPV[None].repeat(B, 1)
            if a.idle_head > 0:
                idl = ((e[:, 3] < 0.5) & (e[:, 7] < 0.5))[:, None]
                tv[:, :6] = torch.where(idl, torch.full_like(tv[:, :6], a.idle_head), tv[:, :6])
            if a.speak_head > 0:
                spk = (e[:, 3] > 0.5)[:, None]
                tv[:, :6] = torch.where(spk, torch.full_like(tv[:, :6], a.speak_head), tv[:, :6])
            if RO is not None:
                pres = RO["present"][:, an]
                emul = np.clip((RO["head_speed"][:, an] / max(np.expm1(RXN["ymu"][1]), 0.5)) ** REACT["energy"], 0.6, 1.6)
                emul = np.where(pres, emul, 1.0).astype(np.float32)
                tv[:, :6] = tv[:, :6] * torch.from_numpy(emul)[:, None]
            d = (m + tv.repeat(1, CH) * s * torch.randn(B, CH * Dd, generator=g)).reshape(B, CH, Dd)
            pbk = torch.sigmoid(b)
            if a.blink_saccade != 1.0:      # blinks ride on gaze shifts (real lift ~2): world-gaze jump in this chunk
                PSg = torch.from_numpy(pos_s.astype(np.float32))
                dd = d * dS
                jx = (dd[:, :, 6] * PSg[6] + dd[:, :, 0] * PSg[0]).sum(1); jy = (dd[:, :, 7] * PSg[7] - dd[:, :, 1] * PSg[1]).sum(1)
                pbk = torch.clamp(pbk * torch.where(torch.hypot(jx, jy) >= 8.0, a.blink_saccade, 1.0), max=0.95)
            bk = (torch.rand(B, generator=g) < pbk).float()
            for j in range(CH):
                t = an + 1 + j
                if t >= T: break
                if a.nod:
                    act = NODT >= 0
                    NODOFF[act, t] = [nod_template(NODT[bi], NODA[bi], NOD2[bi]) for bi in np.where(act)[0]]
                    NODT[act] += 1
                    NODT[NODT > (30 if True else 15)] = -1
                y = torch.clamp(y * (1 - LEAKV) + d[:, j] * dS, -5, 5)
                if a.anchor > 0:      # world gaze = eye + head; pull it toward 0 (the partner) through the eyes
                    PS = torch.from_numpy(pos_s.astype(np.float32))
                    anc = a.anchor
                    if RO is not None:
                        am = np.clip(((1 - RO["avert"][:, t]) / max(1 - RXB["avert"], 0.05)) ** (2 * REACT["gaze"]), 0.3, 2)
                        anc = torch.from_numpy((a.anchor * np.where(RO["present"][:, t], am, 1.0)).astype(np.float32))
                    wx = y[:, 6] * PS[6] + y[:, 0] * PS[0]; wy = y[:, 7] * PS[7] - y[:, 1] * PS[1]
                    y[:, 6] = y[:, 6] - anc * wx / PS[6]; y[:, 7] = y[:, 7] - anc * wy / PS[7]
                Y[:, t] = y.numpy()
                if j == 0: BL[:, t] = bk.numpy()
            clock = torch.where(bk > 0, torch.zeros(B), torch.clamp(clock + CH / 30 / 4, max=1.0))
    if a.nod:      # the nod template rides on the OUTPUT pitch (not the network's own state)
        Y[:, :, 1] += NODOFF / pos_s[1]
    if RO is not None:   # mirroring: expected smile reaction added on the OUTPUT smile
        si = OUT.index("smile")
        Y[:, :, si] += np.where(RO["present"], REACT["smile"] * np.maximum(-0.1, RO["smile_rel"]), 0) / pos_s[si]
    return Y, BL


# ---- held-out segments
groups = {}
for i, (s0, s1) in enumerate(D["seg"]):
    v = int(D["segvid"][i]); p = int(D["segperson"][i])
    if a.people == "val" and meta[v]["id"] not in val_ids:
        continue
    if a.people == "all" and len(groups) >= 400 and (v, p) not in groups:
        continue
    if (a.subset == "dyad" or a.subset_people == "dyad") and D["pf"][s0:s1, 0].mean() < 0.5:
        continue
    groups.setdefault((v, p), []).append((s0, min(s1, s0 + a.maxlen)))
allsegs = [s for g in groups.values() for s in g]
T = max(s1 - s0 for s0, s1 in allsegs)
ext = np.zeros((len(allsegs), T, ext_all.shape[1]), np.float32)
y0 = np.zeros((len(allsegs), len(OUT)), np.float32)
for i, (s0, s1) in enumerate(allsegs):
    ext[i, :s1 - s0] = ext_all[s0:s1]
    m0 = np.nan_to_num(D["M"][s0, cols]); m0[OUT.index("arm")] = nz["arm_fill"] if not np.isfinite(D["M"][s0, COLS.index("arm")]) else m0[OUT.index("arm")]
    y0[i] = np.clip((m0 - pos_c) / pos_s, -5, 5)
if a.zero_pf:
    ext[:, :, 8:17] = 0.0
Yg, Bg = generate(ext, y0, a.temp, a.seed)
def smooth_out(Yx):
    Z = Yx.copy()
    if a.smooth > 0:
        al = 1 - np.exp(-2 * np.pi * a.smooth / 30)
        for t in range(1, Yx.shape[-2]):
            Z[..., t, :6] = Z[..., t - 1, :6] + al * (Yx[..., t, :6] - Z[..., t - 1, :6])
    if a.smile_hz > 0:
        si = OUT.index("smile"); al = 1 - np.exp(-2 * np.pi * a.smile_hz / 30)
        for t in range(1, Yx.shape[-2]):
            Z[..., t, si] = Z[..., t - 1, si] + al * (Yx[..., t, si] - Z[..., t - 1, si])
    return Z
Yg = smooth_out(Yg)


def seg_dict(s0, s1, Mh, blink, bdur):
    selfsp = D["sv"][s0:s1, 3] > 0
    return dict(M=Mh, blink=blink, blinkdur=bdur, st=D["st"][s0:s1].astype(int), selfsp=selfsp,
                partsp=D["pv"][s0:s1, 3] > 0, f0=D["sraw"][s0:s1, 1] * selfsp,
                pM=D["pM"][s0:s1] if np.isfinite(D["pM"][s0:s1, 12]).any() else None)


real_people, gen_people = [], []
i = 0
for key, segs in groups.items():
    rs, gs = [], []
    for s0, s1 in segs:
        L = s1 - s0
        rs.append(seg_dict(s0, s1, D["M"][s0:s1], D["blink"][s0:s1], D["blinkdur"][s0:s1]))
        Mh = np.full((L, len(COLS)), np.nan, np.float32)
        Mh[:, cols] = Yg[i, :L] * pos_s + pos_c
        Mh[:, COLS.index("blink_l")] = 0; Mh[:, COLS.index("blink_r")] = 0
        gs.append(seg_dict(s0, s1, Mh, Bg[i, :L], Bg[i, :L] * 0.2))
        i += 1
    real_people.append(mstats.person_stats(rs)); gen_people.append(mstats.person_stats(gs))
R = mstats.aggregate(real_people); G = mstats.aggregate(gen_people)

KEYS = ["head_yaw_sd10s", "head_pitch_sd10s", "head_roll_sd10s", "head_yaw_speed_med", "head_pitch_speed_med",
        "head_roll_speed_med", "head_pitch_speed_p90", "head_pitch_psd_0.1_0.5Hz", "head_pitch_psd_1.0_2.0Hz",
        "head_pitch_psd_4.0_8.0Hz", "head_yaw_psd_1.0_2.0Hz", "head_tx_sd10s_cm", "head_tz_sd10s_cm", "stillness_frac",
        "nod_per_min", "bignod_per_min", "blink_per_min", "gshift_per_min", "avert_frac", "avert_dur_s_med",
        "smile_sd", "smile_per_min", "brow_sd", "brow_per_min", "arm_speed_med"]
rows = []; errs = []
for st in ("speaking", "listening", "idle"):
    for k in KEYS:
        r = R.get(st, {}).get(k); g = G.get(st, {}).get(k)
        if r is None or g is None:
            continue
        if "psd" in k:
            e = abs(g["med"] - r["med"]) * np.log(10) / 2      # log10 power -> log amplitude ratio
        else:
            e = abs(np.log((g["med"] + 1e-3) / (r["med"] + 1e-3)))
        errs.append(e)
        rows.append(dict(state=st, metric=k, real=r["med"], real_iqr=[r["q25"], r["q75"]], gen=g["med"],
                         gen_iqr=[g["q25"], g["q75"]], log_err=float(e),
                         inside_iqr=bool(r["q25"] <= g["med"] <= r["q75"])))
cp = []
for k in ("blink_after_speech_onset_lift", "blink_at_gaze_shift_lift", "brow_at_accent_lift", "nod_at_accent_lift",
          "nod_after_partner_pause_lift", "bignod_after_partner_pause_lift", "bignod_at_accent_lift"):
    r = R["coupling"].get(k); g = G["coupling"].get(k)
    if r and g:
        cp.append(dict(metric=k, real=r["med"], gen=g["med"]))
res_curves = dict(real_nod_hist=R["coupling"].get("nod_offset_hist"), gen_nod_hist=G["coupling"].get("nod_offset_hist"),
                  real_bignod_hist=R["coupling"].get("bignod_offset_hist"), gen_bignod_hist=G["coupling"].get("bignod_offset_hist"),
                  real_smile_xcorr=R["coupling"].get("smile_xcorr_mean_curve"), gen_smile_xcorr=G["coupling"].get("smile_xcorr_mean_curve"))
for k in ("smile_xcorr_mean_curve", "nod_offset_hist", "bignod_offset_hist"):
    r = R["coupling"].get(k); g = G["coupling"].get(k)
    if r and g:
        if k == "smile_xcorr_mean_curve":
            cp.append(dict(metric="smile_mirror_peak_lag_s", real=r["peak_lag_s"], gen=g["peak_lag_s"]))
            cp.append(dict(metric="smile_mirror_peak_r", real=r["peak_r"], gen=g["peak_r"]))
        else:
            cp.append(dict(metric=k.replace("_hist", "_median_s"), real=r["median_s"], gen=g["median_s"]))

# ---- 10-minute stability: cycle a real speaking stretch, a listening stretch and silence
spk = [s for s in allsegs if D["sv"][s[0]:s[1], 3].mean() > 0.6]
lis = [s for s in allsegs if D["pv"][s[0]:s[1], 3].mean() > 0.5]
pieces = []
rng = np.random.default_rng(0)
while sum(len(p) for p in pieces) < 18000:
    for pool in (spk, lis, None):
        if pool:
            s0, s1 = pool[rng.integers(len(pool))]
            pieces.append(ext_all[s0:min(s1, s0 + 600)])
        else:
            z = np.zeros((300, ext_all.shape[1]), np.float32); pieces.append(z)
long = np.concatenate(pieces)[:18000][None]
YL, BLL = generate(long, np.zeros((1, len(OUT)), np.float32), a.temp, a.seed + 7)
YL = smooth_out(YL)
ML = YL[0] * pos_s + pos_c
stab = {}
for j, c in enumerate(OUT[:8]):
    w = ML[:, j].reshape(-1, 1800)        # 1-minute windows
    stab[c] = dict(max_abs=float(np.abs(ML[:, j]).max()), minute_means=[float(x) for x in w.mean(1)],
                   minute_sd=[float(x) for x in w.std(1)])
stab["blinks_per_min"] = float(BLL.sum() / 10)
res = dict(curves=res_curves, tag=a.tag, temp=a.temp, subset=a.subset, n_people=len(groups), mean_log_err=float(np.mean(errs)),
           frac_inside_iqr=float(np.mean([r["inside_iqr"] for r in rows])), rows=rows, couplings=cp, stability=stab)
outp = Path(a.out) if a.out else md / f"{a.tag}_eval_t{a.temp}{'_' + a.chtemp if a.chtemp else ''}{'_L' + a.leak if a.leak else ''}{'_S' + str(a.smooth) if a.smooth else ''}.json"
res.update(rx=a.rx, react=a.react, chtemp=a.chtemp, leak=a.leak, smooth=a.smooth, nod=a.nod, nodgain=a.nodgain, zero_pf=a.zero_pf,
           people=a.people, subset_people=a.subset_people)
outp.write_text(json.dumps(res, indent=1))
print(f"{a.tag} temp={a.temp} people={len(groups)} mean|log err|={np.mean(errs):.3f} inside-IQR={res['frac_inside_iqr']:.2f}")
for r in rows:
    print(f"  {r['state']:9s} {r['metric']:26s} real {r['real']:8.3g} [{r['real_iqr'][0]:.3g},{r['real_iqr'][1]:.3g}]  gen {r['gen']:8.3g}  {'ok' if r['inside_iqr'] else '--'}")
for c in cp:
    print(f"  coupling {c['metric']:32s} real {c['real']:.3g}  gen {c['gen']:.3g}")
for c in OUT[:8]:
    s = stab[c]; print(f"  stability {c:6s} max|x| {s['max_abs']:.1f}  minute means {np.round(s['minute_means'], 1)}")
print("  stability blinks/min", stab["blinks_per_min"])
