"""the adaptive quality controller of the real-time engine — quality drops before frames do.

The engine holds a ladder of quality tiers (cheapest .. best: the character's internal resolution, and on the lowest
tier no wide occlusion taps).  The controller reads every frame's measured time (render + encode, ms) and picks the
tier for the next frame:

  DOWN, fast   the recent frames are over the budget (mean of the last `down_n`, `down_k` of the last `down_m`, or
               three in a row 12 % over):
               one tier down at once, again after `down_hold` frames if still over.  One slow frame alone (a GC pause,
               a busy CPU) moves nothing; one frame over the hard limit (a visible stutter) with a slow neighbour does.
  UP, slow     after `up_hold` frames (10 s) on a tier, when the 95th percentile of the window, scaled to the next
               tier's cost (what that tier measured last time, else its pixel ratio), fits `up_frac` of the budget.
               A step up that had to be taken back within `fail_window` (60 s) backs off: that tier is not tried
               again for `backoff0` (30 s), then 2x as long each time, up to `backoff_max` (10 min) — on a boundary
               load the controller settles on the lower tier instead of flip-flopping (pass 2: it had gone
               q60 <-> q68 14 times in 4 minutes on a loaded machine).

Pure Python, no GL: engine/rt_engine.py applies the tier; tools/adaptive_test.py exercises it on synthetic loads.
"""
from collections import deque


class AdaptiveTier:
    def __init__(self, costs, start=None, budget_ms=33.3, hard_ms=41.7, fps=24, down_frac=0.96, up_frac=0.90,
                 down_n=6, down_k=4, down_m=12, down_hold=8, up_hold=None, window=None, backoff_max=None,
                 fixed_frac=0.45, warm=24, fail_window=None, backoff0=None):
        """costs: relative cost of each tier, cheapest first (e.g. the squared internal scales)."""
        self.costs = [float(c) for c in costs]
        self.n = len(self.costs)
        self.tier = self.n - 1 if start is None else max(0, min(self.n - 1, int(start)))
        self.budget, self.hard = float(budget_ms), float(hard_ms)
        self.down_frac, self.up_frac = down_frac, up_frac
        self.down_n, self.down_k, self.down_m, self.down_hold = down_n, down_k, down_m, down_hold
        self.up_hold = int(up_hold if up_hold is not None else 10*fps)
        self.window = int(window if window is not None else self.up_hold)     # the whole hold: a load that comes and goes is seen
        self.backoff_max = int(backoff_max if backoff_max is not None else 600*fps)
        self.fail_window = int(fail_window if fail_window is not None else 60*fps)
        self.backoff0 = int(backoff0 if backoff0 is not None else 30*fps)
        self.warm = int(warm)                 # frames ignored at the start (first-use compiles, a cold encoder)
        self.fixed_frac = fixed_frac          # share of a frame that does not scale with the tier (background, post, encode)
        self.hist = deque(maxlen=self.window)
        self.since = 0                        # frames on this tier
        self.frame = 0
        self.seen = {}                        # tier -> (mean ms, frame) the last time it ran
        self.block = {}                       # tier -> frame before which it is not tried again
        self.backoff = {}                     # tier -> current back-off (frames)
        self.up_at = None                     # frame of the last step up (to detect a step taken back)
        self.switches = 0
        self.log = []                         # (frame, from, to, why)

    # ---- helpers
    def _predict(self, to, p95):
        """This tier's p95 scaled to tier `to`."""
        s = self.seen.get(to)
        cur = self.seen.get(self.tier)
        if s is not None and cur is not None and self.frame - s[1] < 12*24 and cur[0] > 1e-3:
            return p95*s[0]/cur[0]
        r = self.costs[to]/max(self.costs[self.tier], 1e-6)
        return p95*(self.fixed_frac + (1.0 - self.fixed_frac)*r)

    def _switch(self, to, why):
        self.log.append((self.frame, self.tier, to, why))
        if len(self.log) > 400:
            del self.log[:200]
        if to < self.tier and self.up_at is not None and self.frame - self.up_at < self.fail_window:
            # the step up did not hold: back off from that tier
            b = min(self.backoff.get(self.tier, self.backoff0//2)*2, self.backoff_max)
            self.backoff[self.tier] = b
            self.block[self.tier] = self.frame + b
        if to > self.tier:
            self.up_at = self.frame
        self.tier = to
        self.hist.clear()
        self.since = 0
        self.switches += 1

    # ---- per frame
    def note(self, total_ms):
        """Feed one frame's render + encode time; returns the tier for the next frame."""
        self.frame += 1
        if self.frame <= self.warm:
            return self.tier
        self.since += 1
        if self.since <= 2:                   # the first frames after a switch carry the switch itself
            return self.tier
        h = self.hist
        h.append(float(total_ms))
        lim = self.budget*self.down_frac
        if len(h) >= 5:
            self.seen[self.tier] = (sorted(h)[len(h)//2], self.frame)       # the median: one spike is not the tier's cost
        if self.tier > 0 and (self.since > self.down_hold or self.switches == 0):
            last = list(h)[-self.down_m:]
            over = sum(1 for x in last if x > lim)
            recent = last[-self.down_n:]
            # (the mean, with most of the frames near the limit: two lone spikes close together are not a load)
            mean_over = len(recent) >= self.down_n and sum(recent)/len(recent) > lim and sorted(recent)[len(recent)//2] > 0.94*lim
            hard = len(last) >= 2 and last[-1] > self.hard and sorted(last[-3:])[len(last[-3:])//2] > lim
            burst = len(last) >= 3 and min(last[-3:]) > 1.12*lim          # three in a row well over: do not wait for the mean
            if mean_over or over >= self.down_k or hard or burst:
                # far over: skip a tier
                step = 2 if (len(recent) >= 3 and sorted(recent[-3:])[1] > 1.45*self.budget and self.tier > 1) else 1      # the median: one spike is not 'far over'
                self._switch(self.tier - step, "over budget")
                return self.tier
        if self.tier < self.n - 1 and self.since >= self.up_hold and len(h) >= min(self.window, self.up_hold) - 2:
            to = self.tier + 1
            if self.frame >= self.block.get(to, 0):
                s = sorted(h)
                p95 = s[min(len(s) - 1, int(0.95*len(s)))]
                if self._predict(to, p95) < self.up_frac*self.budget:
                    self._switch(to, "headroom")
                    return self.tier
        # a tier that has held for a long time forgets its back-off
        if self.since == 2*self.backoff_max:
            self.backoff.pop(self.tier + 1, None)
        return self.tier
