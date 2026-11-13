#!/usr/bin/env python3
"""objects-eval — deterministic STATE checks from the recorded rig columns (+ the engine's own state).

Geometry is general, read from the candidate's renderer/animals/muse.props.json: a FREE prop's world position = its home
pos + (prop{k}_x, _y, _z) (the columns' contract: free = offset from home, world axes); attached = prop{k}_hl / _hr (held)
or _hh (on the head). Half-heights come from the home scene (home y minus the top of what it rests on). Relations:
  on(X, Y)     X free, over Y's top surface (afford.surface size, else its size) and resting at Y's top (bottom within
               -0.03 .. +0.12 H of it)
  in(X, Y)     X free, inside Y's afford.container box
  ...
Per-case checks live in CHECKS (written from each case's `expect`; eval-side, never engine code). A check list ->
{"status": ok|fail|n/a, "items": [{check, ok, detail}]}.
"""
import json, math
from pathlib import Path

ATT = 0.5


class Scene:
    def __init__(self, src, rig_path, state=None):
        P = json.loads((Path(src) / "renderer/animals/muse.props.json").read_text())
        self.props = {p["id"]: p for p in P["props"]}
        self.places = {p["id"]: p for p in P.get("places") or []}
        lines = Path(rig_path).read_text().splitlines()
        self.names = json.loads(lines[0])["rig_params"]
        self.ci = {n: i for i, n in enumerate(self.names)}
        self.rows = [json.loads(x)["a"] for x in lines[1:] if x.strip()]
        self.state = state or {}
        self._hh = {}

    def col(self, n, f=-1):
        i = self.ci.get(n)
        return 0.0 if i is None else float(self.rows[f][i] or 0.0)

    def slot(self, pid):
        return self.props[pid]["slot"]

    def held(self, pid, f=-1):
        k = self.slot(pid)
        return self.col(f"prop{k}_hl", f) > ATT or self.col(f"prop{k}_hr", f) > ATT

    def held_both(self, pid, f=-1):
        k = self.slot(pid)
        return self.col(f"prop{k}_hl", f) > ATT and self.col(f"prop{k}_hr", f) > ATT

    def on_head(self, pid, f=-1):
        return self.col(f"prop{self.slot(pid)}_hh", f) > ATT

    def worn(self, pid, f=-1):
        w = self.state.get("wearing") or []
        return self.on_head(pid, f) or pid in w

    def attached(self, pid, f=-1):
        return self.held(pid, f) or self.on_head(pid, f)

    def home(self, pid):
        return self.props[pid]["home"]["pos"]

    def world(self, pid, f=-1):
        if self.attached(pid, f):
            return None
        k, h = self.slot(pid), self.home(pid)
        return [h[0] + self.col(f"prop{k}_x", f), h[1] + self.col(f"prop{k}_y", f), h[2] + self.col(f"prop{k}_z", f)]

    def offset(self, pid, f=-1):
        k = self.slot(pid)
        return [self.col(f"prop{k}_{a}", f) for a in ("x", "y", "z", "rx", "ry", "rz")]

    def s(self, pid, f=-1):
        return self.col(f"prop{self.slot(pid)}_s", f)

    def half_h(self, pid):
        if pid in self._hh:
            return self._hh[pid]
        p = self.props[pid]
        h = p["home"]
        on = h.get("on")
        if on and on in self.props:
            top = self.home(on)[1] + self.top_off(on)
        else:
            pl = self.places.get(h.get("place"), {})
            top = float(pl.get("top_y", 0.0) or 0.0)
        v = max(0.005, h["pos"][1] - top)
        self._hh[pid] = v
        return v

    def top_off(self, pid):
        a = self.props[pid].get("afford") or {}
        if a.get("surface"):
            return float(a["surface"].get("top_y", 0.0))
        return self.half_h(pid)

    def footprint(self, pid):
        a = self.props[pid].get("afford") or {}
        if a.get("surface") and a["surface"].get("size"):
            w, d = a["surface"]["size"][:2]
            return w / 2, d / 2
        if a.get("container") and a["container"].get("inner"):
            i = a["container"]["inner"]
            return i[0] * 1.3, i[2] * 1.3
        dims = self.props[pid].get("dims")
        if dims:
            return dims[0] / 2, dims[-1] / 2
        r = float(self.props[pid].get("size") or 0.05) * 0.6
        return r, r

    # ---------------------------------------------------------------- relations (final frame unless f given)
    def on(self, x, y, f=-1):
        X, Y = self.world(x, f), self.world(y, f)
        if X is None or Y is None:
            return False, f"{x} {'attached' if X is None else 'free'}, {y} {'attached' if Y is None else 'free'}"
        hx, hz = self.footprint(y)
        dx, dz = X[0] - Y[0], X[2] - Y[2]
        gap = (X[1] - self.half_h(x)) - (Y[1] + self.top_off(y))
        ok = abs(dx) <= hx + 0.04 and abs(dz) <= hz + 0.04 and -0.03 <= gap <= 0.12
        return ok, f"dx {dx:+.3f} dz {dz:+.3f} (half {hx:.3f}x{hz:.3f}), gap above top {gap:+.3f} H"

    def inside(self, x, y, f=-1):
        X, Y = self.world(x, f), self.world(y, f)
        if X is None or Y is None:
            return False, f"{x} {'attached' if X is None else 'free'}, {y} {'attached' if Y is None else 'free'}"
        c = (self.props[y].get("afford") or {}).get("container") or {}
        inner = c.get("inner") or [0.05, 0.05, 0.05]
        fy = float(c.get("floor_y", -inner[1]))
        dx, dz, dy = X[0] - Y[0], X[2] - Y[2], X[1] - Y[1]
        ok = abs(dx) <= inner[0] + 0.02 and abs(dz) <= inner[2] + 0.02 and fy - 0.02 <= dy - 0.0 <= fy + 2 * inner[1] + self.half_h(x) + 0.03
        return ok, f"dx {dx:+.3f} dz {dz:+.3f} dy {dy:+.3f} (inner {inner}, floor_y {fy})"

    def on_floor(self, x, f=-1):
        X = self.world(x, f)
        if X is None:
            return False, f"{x} attached"
        b = X[1] - self.half_h(x)
        return b < 0.04, f"bottom {b:+.3f} H"

    def at_home(self, x, f=-1, tol=0.03):
        if self.attached(x, f):
            return False, "attached"
        o = self.offset(x, f)
        d = math.sqrt(o[0] ** 2 + o[1] ** 2 + o[2] ** 2)
        r = max(abs(((v + 0.5) % 1) - 0.5) for v in o[3:])
        return d <= tol and r <= 0.03, f"offset {d:.3f} H, rot {r:.3f} turns"

    def ever(self, fn, frames=None):
        for f in frames if frames is not None else range(len(self.rows)):
            if fn(f):
                return True, f"frame {f}"
        return False, "never"

    def flew(self, x, frames):
        """free and above its resting height by > 0.15 H at some frame while not attached (tossed / falling)"""
        best = -9
        for f in frames:
            W = self.world(x, f)
            if W is not None:
                best = max(best, W[1] - self.half_h(x))
        return best > 0.15, f"max free bottom height {best:+.3f} H"


def run_checks(src, case_dir, checks, main_frame0=0):
    cj = json.loads((Path(case_dir) / "case.json").read_text())
    S = Scene(src, Path(case_dir) / "rig.jsonl", cj.get("state"))
    frames = range(max(0, main_frame0), len(S.rows))
    items = []
    if not checks:
        return {"status": "n/a", "items": []}
    for ch in checks:
        op, a = ch[0], ch[1:]
        try:
            if op == "worn":
                ok, d = S.worn(a[0]), f"hh {S.col(f'prop{S.slot(a[0])}_hh'):.2f}, state wearing {S.state.get('wearing')}"
            elif op == "not_worn":
                ok, d = not S.worn(a[0]), f"hh {S.col(f'prop{S.slot(a[0])}_hh'):.2f}"
            elif op == "open":
                ok, d = S.s(a[0]) > 0.5 or a[0] in (S.state.get("open") or []), f"s {S.s(a[0]):.2f}"
            elif op == "closed":
                ok, d = S.s(a[0]) < 0.2, f"s {S.s(a[0]):.2f}"
            elif op == "on":
                ok, d = S.on(a[0], a[1])
            elif op == "not_on":
                ok, d = S.on(a[0], a[1]); ok = not ok
            elif op == "on_any_order":
                ok1, d1 = S.on(a[0], a[1]); ok2, d2 = S.on(a[1], a[0]); ok, d = ok1 or ok2, f"{a[0]} on {a[1]}: {d1}; {a[1]} on {a[0]}: {d2}"
            elif op == "in":
                ok, d = S.inside(a[0], a[1])
            elif op == "on_head":
                ok, d = S.on_head(a[0]) and not S.held(a[0]), f"hh {S.col(f'prop{S.slot(a[0])}_hh'):.2f}, held {S.held(a[0])}"
            elif op == "held_end":
                ok, d = S.held(a[0]), "held at the end" if S.held(a[0]) else "not held at the end"
            elif op == "not_held_end":
                ok, d = not S.held(a[0]), "held at the end" if S.held(a[0]) else "not held at the end"
            elif op == "held_ever":
                ok, d = S.ever(lambda f: S.held(a[0], f), frames)
            elif op == "held_both_ever":
                ok, d = S.ever(lambda f: S.held_both(a[0], f), frames)
            elif op == "flew":
                ok, d = S.flew(a[0], frames)
            elif op == "on_floor":
                ok, d = S.on_floor(a[0])
            elif op == "moved":
                ok, d = S.at_home(a[0]); ok = not ok
            elif op == "home":
                ok, d = S.at_home(a[0])
            elif op == "all_home":
                bad = [p for p in S.props if p not in ("desk", "chair") and not S.at_home(p)[0]]
                ok, d = not bad, "not home: " + ", ".join(bad) if bad else "all props home"
            elif op == "unchanged":
                bad = [p for p in S.props if not S.at_home(p)[0]]
                ok, d = not bad, "changed: " + ", ".join(bad) if bad else "nothing moved"
            elif op == "ever_s":
                ok, d = S.ever(lambda f: S.s(a[0], f) > float(a[1]), frames)
            elif op == "spun":
                k = S.slot(a[0]); tot = 0.0; prev = None
                for f in frames:
                    v = S.col(f"prop{k}_ry", f)
                    if prev is not None:
                        dv = v - prev; dv -= round(dv); tot += abs(dv)
                    prev = v
                ok, d = tot > 0.75, f"total |ry| travel {tot:.2f} turns"
            elif op == "rotated":
                r = S.offset(a[0])[4]; dr = abs(((r - float(a[1]) + 0.5) % 1) - 0.5)
                ok, d = dr < 0.1 and S.world(a[0]) is not None, f"ry offset {r:+.2f} turns (want {a[1]})"
            elif op == "near":
                X, Y = S.world(a[0]), S.world(a[1])
                if X is None or Y is None:
                    ok, d = False, "attached"
                else:
                    dd = math.hypot(X[0] - Y[0], X[2] - Y[2]); hx, hz = S.footprint(a[1])
                    ok, d = max(hx, hz) * 0.8 < dd < max(hx, hz) + 0.35, f"distance {dd:.3f} H"
            elif op == "under":
                X, Y = S.world(a[0]), S.world(a[1])
                if X is None or Y is None:
                    ok, d = False, "attached"
                else:
                    dims = S.props[a[1]].get("dims") or [0.3, 0.3, 0.3]
                    dx, dz = X[0] - Y[0], X[2] - Y[2]
                    ok = abs(dx) < dims[0] / 2 and abs(dz) < dims[-1] / 2 and X[1] < Y[1] + 0.1
                    d = f"dx {dx:+.3f} dz {dz:+.3f} y {X[1]:.3f}"
            elif op == "behind":
                X, Y = S.world(a[0]), S.world(a[1])
                if X is None or Y is None:
                    ok, d = False, "attached"
                else:
                    dx, dz = X[0] - Y[0], X[2] - Y[2]
                    ok, d = dz < -0.05 and abs(dx) < 0.2, f"dx {dx:+.3f} dz {dz:+.3f}"
            elif op == "seated":
                ok, d = S.state.get("posture") == "seated" or bool(S.state.get("sittingOn")), f"posture {S.state.get('posture')}"
            else:
                ok, d = False, "unknown check"
        except Exception as e:  # noqa: BLE001
            ok, d = False, f"check error {e!r}"
        items.append({"check": ch, "ok": bool(ok), "detail": d})
    return {"status": "ok" if all(i["ok"] for i in items) else "fail", "items": items}


# per case, from requests.json `expect` (only where expect implies a state; [] = judge only)
CHECKS = {
    "o01": [["worn", "hat"]],
    "o02": [["worn", "glasses"]],
    "o03": [["open", "gift_box"], ["not_on", "glasses", "gift_box"]],
    "o04": [["in", "block_a", "gift_box"], ["open", "gift_box"]],
    "o05": [["ever_s", "watering_can", 0.3], ["not_held_end", "watering_can"]],
    "o06": [["on", "mug", "book"]],
    "o07": [["on", "hat", "plant"]],
    "o08": [["on_head", "ball"]],
    "o09": [["flew", "ball"], ["held_end", "ball"]],
    "o10": [],
    "o11": [["flew", "block_a"], ["flew", "block_b"]],
    "o12": [["held_ever", "book"], ["spun", "book"]],
    "o13": [["held_end", "block_a"]],
    "o14": [["held_end", "block_b"]],
    "o15": [["flew", "ball"]],
    "o16": [["all_home"], ["not_worn", "hat"]],
    "o17": [["on", "block_b", "gift_box"], ["on", "block_a", "block_b"]],
    "o18": [["unchanged"]],
    "o19": [["near", "block_a", "laptop"], ["on_floor", "block_a"]],
    "o20": [["under", "ball", "desk"]],
    "o21": [["behind", "block_b", "gift_box"]],
    "o22": [],
    "o23": [],
    "o24": [["unchanged"]],
    "o25": [["unchanged"]],
    "o26": [["unchanged"]],
    "o27": [["flew", "ball"], ["on_floor", "ball"], ["not_held_end", "ball"]],
    "o28": [["flew", "block_b"]],
    "o29": [["held_end", "book"], ["moved", "mug"], ["not_held_end", "mug"]],
    "o30": [["not_on", "block_a", "book"], ["moved", "block_a"]],
    "o31": [["held_ever", "hat"]],
    "o32": [["worn", "hat"], ["worn", "glasses"]],
    "o33": [["in", "hat", "gift_box"], ["in", "glasses", "gift_box"], ["not_worn", "hat"], ["not_worn", "glasses"]],
    "o34": [["closed", "gift_box"]],
    "o35": [["held_ever", "watering_can"]],
    "o36": [["held_both_ever", "gift_box"]],
    "o37": [["on", "ball", "mug"]],
    "o38": [["held_ever", "mug"]],
    "o39": [["worn", "headphones"], ["seated"]],
    "o40": [["worn", "headphones"]],
    "o41": [["on_any_order", "block_a", "block_b"], ["moved", "block_a"], ["moved", "block_b"]],
    "o42": [["not_on", "block_b", "block_a"], ["not_on", "block_a", "block_b"]],
    "o43": [["moved", "ball"], ["not_held_end", "ball"]],
    "o44": [["moved", "block_a"], ["not_held_end", "block_a"]],
    "o45": [["rotated", "gift_box", 0.5]],
    "o46": [["on", "glasses", "hat"]],
    "o47": [["all_home"], ["closed", "gift_box"]],
}
