import json,glob,os,sys
# usage: rt_table.py [<rt dir>]   (default $MM_WORK_DIR/runs/p2react/rt)
O=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.environ.get("MM_WORK_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "work")), "runs/p2react/rt")
for f in ["A3"]+sorted(os.path.basename(p)[:-5] for p in glob.glob(O+"/*.json") if not p.endswith("_react.json") and "lane" not in p):
    rp=O+f"/{f}_react.json" if f!="A3" else O+"/A3_react.json"
    if not os.path.exists(rp): continue
    r=json.load(open(rp))["gen"]
    if f=="A3": R="9.0"; sm="36.0"
    else:
        j=json.load(open(O+f"/{f}.json")); R=f"{j['realism']:.2f}"; sm=f"{j['gen']['smiles_per_min']:.1f}/{j['events']['smile']['fired_all']:.0f}fired"
    print(f"{f:16} real {R:6} smi>smi {r['smile->smile']['excess']:+.3f} (c{r['smile->smile']['chance']:.2f}) nod {r['nod->nod']['excess']:+.3f} lau>smi {r['laugh->smile']['excess']:+.3f} sm/min {r['smiles_per_min']:.1f} {sm}")
