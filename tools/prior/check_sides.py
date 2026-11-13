"""Verify MediaPipe side + sign conventions against raw landmarks on extracted clips."""
import sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from motionlib import *
for f in sys.argv[1:]:
    d = np.load(f)
    for s in range(len(d["streams"])):
        p = d["present"][s].astype(bool); lm = d["lmk"][s][p]; bl = d["blend"][s][p]
        if p.sum() < 100: continue
        def op(t, b, o, i): return (lm[:, LM[b], 1] - lm[:, LM[t], 1]) / (np.abs(lm[:, LM[o], 0] - lm[:, LM[i], 0]) + 1e-6)
        oSL = op("eR_top", "eR_bot", "eR_out", "eR_in")    # landmarks 159/145/33/133 = subject right eye
        oSR = op("eL_top", "eL_bot", "eL_out", "eL_in")
        print("screen-left eye x (should be < screen-right):", lm[:, LM["eR_out"], 0].mean().round(3), lm[:, LM["eL_out"], 0].mean().round(3))
        dl = bl[:, B["eyeBlinkLeft"]] - bl[:, B["eyeBlinkRight"]]
        print("corr(blinkLeft-blinkRight, open_screenL-open_screenR) =", np.corrcoef(dl, oSL - oSR)[0, 1].round(3), "(+ => MP Left = screen RIGHT)")
        yaw, pitch, roll, t = euler_from_mat(d["mat"][s][p])
        em = (lm[:, LM["eR_out"], :2] + lm[:, LM["eL_out"], :2]) / 2
        ew = np.abs(lm[:, LM["eR_out"], 0] - lm[:, LM["eL_out"], 0])
        print("corr(yaw, nose_x - eyemid_x)=", np.corrcoef(yaw, (lm[:, 0, 0] - em[:, 0]) / ew)[0, 1].round(3), "(want +)")
        print("corr(pitch, nose_y - eyemid_y)=", np.corrcoef(pitch, (lm[:, 0, 1] - em[:, 1]) / ew)[0, 1].round(3), "(want +)")
        ang = np.degrees(np.arctan2(lm[:, LM["eL_out"], 1] - lm[:, LM["eR_out"], 1], lm[:, LM["eL_out"], 0] - lm[:, LM["eR_out"], 0]))
        print("corr(roll, eye-line angle (image y down; + = right eye lower = crown to screen LEFT))=", np.corrcoef(roll, ang)[0, 1].round(3), "(want -)")
        print("tx corr nose x:", np.corrcoef(t[:, 0], lm[:, 0, 0])[0, 1].round(3), " ty corr nose y:", np.corrcoef(t[:, 1], lm[:, 0, 1])[0, 1].round(3), "(want +, -)")
print("--- eyeLook side check over all files (iris x offset vs eyeLookOutLeft-eyeLookInLeft)")
cs = []
for f in sys.argv[1:]:
    d = np.load(f)
    for s in range(len(d["streams"])):
        p = d["present"][s].astype(bool)
        if p.sum() < 100: continue
        T, _, _ = person_table(d, s)
        bl = d["blend"][s][p]
        cs.append(np.corrcoef(T[p, C["gx"]], bl[:, B["eyeLookOutLeft"]] - bl[:, B["eyeLookInLeft"]])[0, 1])
print("median corr:", np.median(cs).round(3), "(+ => MP 'Left' = subject's left = SCREEN RIGHT)", np.round(cs, 2))
