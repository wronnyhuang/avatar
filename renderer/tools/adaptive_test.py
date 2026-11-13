#!/usr/bin/env python3
"""the adaptive tier controller on synthetic loads (no GPU).  python tools/adaptive_test.py -> exit 0 = pass."""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))
from rt_adaptive import AdaptiveTier

SC = [0.42, 0.5, 0.6, 0.75, 0.87, 1.0]
COST = [s*s for s in SC]


def run(load, n, start=3, seed=1, **kw):
    """load(frame) -> ms of the BEST tier's scalable part; frame time = 9 fixed + load * cost + noise."""
    rnd = random.Random(seed)
    c = AdaptiveTier(COST, start=start, **kw)
    over, hard, tiers = 0, 0, []
    for f in range(n):
        ms = 9.0 + load(f)*COST[c.tier] + rnd.gauss(0, 0.6) + (25.0 if rnd.random() < 0.004 else 0.0)   # rare one-frame spikes
        over += ms > 33.3; hard += ms > 41.7
        tiers.append(c.tier)
        c.note(ms)
    return c, over, hard, tiers


fails = []
def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fails.append(name)

# 1. a light load: climbs to the top tier and stays
c, over, hard, t = run(lambda f: 14.0, 24*60)
check("light load reaches the top tier", t[-1] == len(SC) - 1 and c.switches <= 4, f"final {t[-1]} switches {c.switches}")
# 2. a heavy load: comes down within a second, then holds without oscillating
c, over, hard, t = run(lambda f: 62.0, 24*120)
first_ok = next(i for i, x in enumerate(t) if 9.0 + 62.0*COST[x] < 32.0)
check("heavy load drops within 1.5 s", first_ok <= 24 + 36, f"frame {first_ok}")
check("heavy load: no oscillation", c.switches <= 12, f"switches {c.switches} in 120 s")
check("heavy load: frames over 33 ms under 3 %", over < 0.03*len(t), f"{over}/{len(t)}")
# 3. a step up in load mid-session (hands raised by the face), then back: follows both ways
c, over, hard, t = run(lambda f: 34.0 if (f < 24*20 or f > 24*40) else 60.0, 24*80)
mid = t[24*30]; end = t[-1]
check("load step: lower tier during it", mid < t[24*19], f"{t[24*19]} -> {mid}")
check("load step: recovers after it", end >= t[24*19], f"end {end}")
check("load step: stutters (over 41.7 ms) = the injected spikes (~8) + the few frames it takes to react", hard <= 20, f"{hard}")
# 4. single spikes move nothing
c = AdaptiveTier(COST, start=3)
for f in range(24*20):
    c.note(60.0 if f % 97 == 50 else 24.0)
check("lone spikes do not drop the tier", min(x[2] for x in c.log) >= 3 if c.log else True, str(c.log[:3]))
# 5. boundary load: back-off keeps the switch count low
c, over, hard, t = run(lambda f: 43.0, 24*300)
check("boundary load: settles on the tier that fits (at most 8 switches in 300 s, never back up to the one that does not)",
      c.switches <= 8 and not any(x[3] == "headroom" and x[2] > 2 for x in c.log), f"switches {c.switches} {[(x[0]//24, x[1], x[2]) for x in c.log]} in 300 s, over33 {over}")
# 6. a loaded machine: the next tier up fits on paper but not in fact (the load grows with the tier more than its pixels say)
c, over, hard, t = run(lambda f: 40.0 + (12.0 if f % 240 < 150 else 0.0), 24*240)
check("uneven load: at most 6 switches in 4 min", c.switches <= 6, f"switches {c.switches} {[(x[0]//24, x[1], x[2]) for x in c.log]}, over33 {over}/{len(t)}")
sys.exit(1 if fails else 0)
