"""Eval JSONs -> docs/PRIOR_EVAL.md (+ docs/PRIOR_EVAL.json with the raw numbers).
Usage: python -I make_eval_doc.py <models_dir> <docs_dir>"""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from prior_model import RX

md, docs = Path(sys.argv[1]), Path(sys.argv[2])
J = lambda n: json.loads((md / n).read_text()) if (md / n).exists() else None
fin_val, fin_all = J("FINAL_val.json"), J("final_all_rx_s07.json")
dy_rx, dy_norx, dy_zero = J("final_dyadval_rx_s07.json"), J("FINAL_dyadval_norx.json"), J("FINAL_dyadval_zeropf.json")
v1 = J("V1_on_v2split_val.json")
rx, rxa = J("rx3_eval.json"), J("rx_ablate_eval.json")
KEY = ["head_yaw_sd10s", "head_pitch_sd10s", "head_roll_sd10s", "head_yaw_speed_med", "head_pitch_speed_med",
       "head_pitch_psd_1.0_2.0Hz", "head_pitch_psd_4.0_8.0Hz", "head_tx_sd10s_cm", "stillness_frac", "nod_per_min",
       "bignod_per_min", "blink_per_min", "gshift_per_min", "avert_frac", "avert_dur_s_med", "smile_sd", "smile_per_min",
       "brow_sd", "brow_per_min"]


def table(res, title):
    rows = {(r["state"], r["metric"]): r for r in res["rows"]}
    out = [f"### {title}", "", f"{res['n_people']} people; mean |log(generated/real)| = **{res['mean_log_err']:.3f}**; "
           f"generated median inside the real inter-quartile range for **{100 * res['frac_inside_iqr']:.0f}%** of "
           f"{len(res['rows'])} state×metric cells.", "",
           "| metric | speaking real [IQR] | gen | listening real [IQR] | gen | idle real [IQR] | gen |", "|---|---|---|---|---|---|---|"]
    for k in KEY:
        cells = []
        for s in ("speaking", "listening", "idle"):
            r = rows.get((s, k))
            if r is None:
                cells += ["—", "—"]; continue
            cells += [f"{r['real']:.3g} [{r['real_iqr'][0]:.3g}, {r['real_iqr'][1]:.3g}]",
                      f"{r['gen']:.3g}{'' if r['inside_iqr'] else ' ✗'}"]
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out += ["", "| coupling | real | generated |", "|---|---|---|"]
    for c in res["couplings"]:
        out.append(f"| {c['metric']} | {c['real']:.3g} | {c['gen']:.3g} |")
    return "\n".join(out) + "\n"


def stab(res):
    s = res["stability"]
    out = ["| channel | max abs over 10 min | per-minute means |", "|---|---|---|"]
    for c in ("yaw", "pitch", "roll", "tx", "ty", "tz", "gx", "gy"):
        out.append(f"| {c} | {s[c]['max_abs']:.1f} | {' '.join(f'{v:+.1f}' for v in s[c]['minute_means'])} |")
    out.append(f"| blinks / min | {s['blinks_per_min']:.1f} | |")
    return "\n".join(out) + "\n"


def pick(res, state, metric):
    for r in res["rows"]:
        if r["state"] == state and r["metric"] == metric:
            return r
    return None


def cpl(res, m):
    for c in res["couplings"]:
        if c["metric"] == m:
            return c
    return None


# reaction-model probe: partner smiles for 3 s while talking to the (silent) avatar
sd = torch.load(md / "rx3.pt"); rxm = RX(H=sd["gru.weight_hh_l0"].shape[1]); rxm.load_state_dict(sd); rxm.eval()
nm = json.loads((md / "rx3_weights.json").read_text())["norm"]
T = 300; x = np.zeros((1, T, 14), np.float32)
x[0, :, 0] = 1; x[0, :, 1] = 0.05; x[0, :, 3] = 0.05; x[0, :, 9] = 0.6; x[0, :, 11] = 1; x[0, :, 12] = 1
x2 = x.copy(); x2[0, 90:180, 1] = 0.6
with torch.no_grad():
    o1 = rxm(torch.from_numpy(x))[0][0].numpy(); o2 = rxm(torch.from_numpy(x2))[0][0].numpy()
ds = (o2[:, 0] - o1[:, 0]) * nm["ysd"][0]
probe = dict(peak=float(ds.max()), t_peak_s=float((np.argmax(ds) - 90) / 30),
             t_half_s=float((np.argmax(ds >= 0.5 * ds.max()) - 90) / 30), decay_half_s=float((np.argmax((np.arange(T) > np.argmax(ds)) & (ds < 0.5 * ds.max())) - 180) / 30))
x3 = x.copy(); x3[0, 90:, 9:13] = 0.0          # partner stops talking at 3 s
with torch.no_grad():
    o3 = rxm(torch.from_numpy(x3))[0][0].numpy()
sg = lambda v: 1 / (1 + np.exp(-v))
probe_pause = dict(bignod_p_before=float(sg(o3[60:90, 4]).mean()), bignod_p_0_1s_after=float(sg(o3[90:120, 4]).mean()),
                   avert_p_before=float(sg(o3[60:90, 5]).mean()), avert_p_after=float(sg(o3[120:180, 5]).mean()))

md_ = ["# Prior evaluation — generated motion vs real people", "",
       "All numbers from `tools/prior/eval_prior.py`: the network runs FREE (sampling, shipped defaults) on the real live-"
       "available inputs of each person (their own speech, the partner's speech and face), and the same statistics as "
       "docs/MOTION_STATS.md are computed on the generated motion and on the real motion of the same people. "
       "`✗` = generated median outside the real IQR. Held-out = videos never seen in training (video-level split).", ""]
if fin_val: md_.append(table(fin_val, "Held-out people (shipped model: general prior + nod channel + reaction model)"))
if fin_all: md_.append(table(fin_all, "All 171 people (train + held-out; distribution check over the whole population)"))
if v1 and fin_val:
    md_ += ["### v1 (first commit) vs shipped, same held-out people", "",
            "| | mean abs log err | inside IQR | speaking big nods/min (real) | listening big nods/min (real) |", "|---|---|---|---|---|"]
    for name, r in (("v1 (chunk-6, no nod channel, no dyads)", v1), ("shipped v2", fin_val)):
        a_, b_ = pick(r, "speaking", "bignod_per_min"), pick(r, "listening", "bignod_per_min")
        md_.append(f"| {name} | {r['mean_log_err']:.3f} | {100 * r['frac_inside_iqr']:.0f}% | "
                   f"{a_['gen']:.1f} ({a_['real']:.1f}) | {b_['gen']:.1f} ({b_['real']:.1f}) |")
    md_.append("")
if dy_rx and dy_norx:
    md_ += ["## Dyadic (two-person) held-out evaluation", "",
            "People in held-out split-screen calls (both faces visible). The partner's real face/voice drive the model; "
            "the avatar's generated face is compared with the real listener/speaker.", "",
            "| | mean abs log err | smile mirroring peak r (lag) | real | listening big nods/min (real) | listening avert frac (real) | listening stillness (real) |",
            "|---|---|---|---|---|---|---|"]
    for name, r in (("general prior only", dy_norx), ("+ reaction model (shipped)", dy_rx), ("general prior, partner face HIDDEN (ablation)", dy_zero)):
        if not r: continue
        sr, sl = cpl(r, "smile_mirror_peak_r"), cpl(r, "smile_mirror_peak_lag_s")
        bn, av, stl = pick(r, "listening", "bignod_per_min"), pick(r, "listening", "avert_frac"), pick(r, "listening", "stillness_frac")
        md_.append(f"| {name} | {r['mean_log_err']:.3f} | {sr['gen']:.3f} ({sl['gen']:+.1f} s) | {sr['real']:.3f} ({sl['real']:+.1f} s) | "
                   f"{bn['gen']:.1f} ({bn['real']:.1f}) | {av['gen']:.2f} ({av['real']:.2f}) | {stl['gen']:.2f} ({stl['real']:.2f}) |")
    md_ += [""]
    for name, r in (("real", dy_rx["curves"]["real_bignod_hist"]), ("generated", dy_rx["curves"]["gen_bignod_hist"])):
        if r:
            md_.append(f"Big-nod onsets relative to the partner's pause onsets ({name}, n={r['n']}, median {r['median_s']:+.2f} s): "
                       + ", ".join(f"{e:+.2f}s:{c}" for e, c in zip(r["edges_s"][:-1], r["counts"])))
    md_.append("")
if rx and rxa:
    md_ += ["## The reaction model on its own (held-out dyads)", "",
            "Partner-only inputs (face 9 + voice 4) + the self-speaking flag -> this person's next-0.5 s reaction; the "
            "ablation hides the partner (only the self-speaking flag remains).", "",
            "| target | with partner | partner hidden | base rate |", "|---|---|---|---|",
            f"| own smile (r, predicted vs real) | {rx['smile_rel_r']:.3f} | {rxa['smile_rel_r']:.3f} | |",
            f"| own head speed (r) | {rx['head_speed_r']:.3f} | {rxa['head_speed_r']:.3f} | |"]
    for k in rx["auc"]:
        md_.append(f"| {k} (AUC) | {rx['auc'][k]:.3f} | {rxa['auc'][k]:.3f} | {rx['base_rates'][k]:.3f} |")
    md_ += ["", f"Probe (partner smiles 0.6 for 3 s while talking; avatar silent): predicted own smile rises by "
            f"{probe['peak']:.3f} (blendshape units), half-rise {probe['t_half_s']:.2f} s after the partner's smile onset, "
            f"peak at {probe['t_peak_s']:.2f} s. Partner stops talking: P(big nod in next 0.5 s) "
            f"{probe_pause['bignod_p_before']:.3f} -> {probe_pause['bignod_p_0_1s_after']:.3f}; P(gaze averted) "
            f"{probe_pause['avert_p_before']:.3f} -> {probe_pause['avert_p_after']:.3f}.", ""]
if fin_val:
    md_ += ["## Stability (10 minutes free-running: speaking 20 s / listening 20 s / silence 10 s cycles)", "",
            "Human units (deg / cm), before the rig gain.", "", stab(fin_val)]
(docs / "PRIOR_EVAL.md").write_text("\n".join(md_) + "\n")
(docs / "PRIOR_EVAL.json").write_text(json.dumps(dict(final_val=fin_val, final_all=fin_all, dyad_rx=dy_rx, dyad_norx=dy_norx,
                                                      dyad_zero_pf=dy_zero, v1_val=v1, rx=rx, rx_ablate=rxa, rx_probe=probe,
                                                      rx_probe_pause=probe_pause), default=float))
print("\n".join(md_)[:6000])
