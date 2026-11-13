"""Build the STEP0-RT runtime variants from the nxHRC-A3 runtime and print them as JSON lines (tag \t rt)."""
import copy, json, os, sys
T = os.environ.get("MM_WORK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "work"))
base = json.load(open(f"{T}/runs/p2rt/nxHRC-A3/bundle/config.json"))["runtime"]
base.pop("input_gain_db", None)
def v(th=None, anchor=None, tg=None, g=None):
    r = copy.deepcopy(base)
    if th is not None: r["events"]["thresholds"][2] = th
    if anchor == "noSmile": r["anchor"]["cols"] = [c for c in r["anchor"]["cols"] if "Smile" not in c]
    if anchor == "tau12": r["anchor"]["tau_s"] = 12
    if tg: r["temperature_groups"].update(tg)
    if g: r["guidance"] = g
    return r
V = {
 "s10": v(th=0.565), "s15": v(th=0.51),
 "s10_anc": v(th=0.565, anchor="noSmile"),
 "s10_tmp": v(th=0.565, tg={"mouth": 0.1, "face_upper": 0.15}),
 "s10_g3": v(th=0.565, g=3.0),
 "s10_anc_tmp": v(th=0.565, anchor="noSmile", tg={"mouth": 0.1, "face_upper": 0.15}),
 "s10_tau12": v(th=0.565, anchor="tau12"),
 "s10_anc_tmp_g3": v(th=0.565, anchor="noSmile", tg={"mouth": 0.1, "face_upper": 0.15}, g=3.0),
}
if __name__ == "__main__":
    for k in (sys.argv[1:] or V): print(k + "\t" + json.dumps(V[k]))
