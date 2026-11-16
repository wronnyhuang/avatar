"""tests: a tiny FAKE of the app/semantic interface (CONTRACT section 2) — no IK, no rig. Each complete
top-level statement of the fenced python block becomes one batch of dense poses on a fake channel set:
    name(args..., dur=x)      -> poses on 'head_yaw' (and 'mouth_open' for speak-ish names) over dur s (default 0.5)
    look_at('user')           -> head_yaw = targets['user'][0] / 10 (live target: replay with new targets differs)
    slow(...)                 -> sleeps 0.2 s inside feed (off-loop test)
    boom(...)                 -> a statement error (skipped, recorded)
    keep_pose()               -> no return-to-neutral tail
A statement is complete when a later column-0 line has started or the fence closed."""
import ast
import threading
import time

CHARACTERS = ("muse", "fox")
FEED_THREADS = []        # thread names feed ran on (tests check it is never the event loop's)


def system_prompt(character, targets=None):
    return f"FAKE SEMANTIC SYSTEM for {character}; live targets: " + ", ".join(sorted(targets or []))


def user_prompt(description, state=None):
    return f'Command: "{description}"\n'


def resolve_targets(character, state):
    out = {"camera": (0.0, 60.0, 150.0), "user": (0.0, 60.0, 150.0)}
    u = (((state or {}).get("sem") or {}).get("user") or {}) if isinstance(state, dict) else {}
    if u.get("present"):
        out["user"] = (float(u.get("ax", 0)) * 100.0, 60.0 + float(u.get("ay", 0)) * 50.0, 150.0)
    return out


def _code(text):
    i = text.find("```")
    if i < 0:
        return None, False
    j = text.find("\n", i)
    if j < 0:
        return None, False
    body = text[j + 1:]
    k = body.find("```")
    if k >= 0:
        return body[:k], True
    return body, False


class Stream:
    def __init__(self, character, targets=None, seed=0, hz=20, t_origin=0.0):
        self.character, self.targets, self.hz = character, dict(targets or {}), hz
        self.text = ""
        self.done_stmts = 0
        self.t = float(t_origin)
        self.errors, self.warnings, self.exec_ms = [], [], []
        self.keep = False
        self.channels = set()
        self.finished = False
        self.program = ""

    def _stmts(self, final):
        code, closed = _code(self.text)
        if code is None:
            return []
        self.program = code
        lines = code.split("\n")
        starts = [i for i, ln in enumerate(lines) if ln and ln[0] not in " \t#"]
        out = []
        for n, i in enumerate(starts):
            end = starts[n + 1] if n + 1 < len(starts) else None
            if end is None and not (closed or final):
                break
            out.append("\n".join(lines[i:end if end is not None else len(lines)]))
        return out

    def _run(self, src, k):
        t0 = time.perf_counter()
        FEED_THREADS.append(threading.current_thread().name)
        try:
            node = ast.parse(src.strip()).body[0]
            call = node.value if isinstance(node, ast.Expr) else None
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                raise ValueError("not a call")
            name = call.func.id
            dur = 0.5
            for kw in call.keywords:
                if kw.arg == "dur":
                    dur = float(ast.literal_eval(kw.value))
            if name == "boom":
                raise ValueError("boom")
            if name == "slow":
                time.sleep(0.2)
            if name == "keep_pose":
                self.keep = True
                return []
            val = 0.2 * (k + 1)
            if name == "look_at" and call.args:
                tgt = ast.literal_eval(call.args[0])
                if tgt not in self.targets:
                    self.warnings.append(f"unknown target {tgt!r} -> camera")
                    tgt = "camera"
                val = self.targets[tgt][0] / 10.0
            n = max(1, int(round(dur * self.hz)))
            poses = []
            for i in range(1, n + 1):
                p = {"t": round(self.t + i / self.hz, 4), "head_yaw": round(val * i / n, 4)}
                if name in ("speak", "open_mouth"):
                    p["mouth_open"] = 0.5
                    self.channels.add("mouth_open")
                poses.append(p)
            self.channels.add("head_yaw")
            t0b = self.t
            self.t = poses[-1]["t"]
            return [{"stmt": k, "t0": t0b, "t1": self.t, "poses": poses, "final": False}]
        except Exception as e:  # noqa: BLE001
            self.errors.append({"stmt": k, "error": f"{type(e).__name__}: {e}"})
            return []
        finally:
            self.exec_ms.append(round((time.perf_counter() - t0) * 1000, 2))

    def _advance(self, final):
        out = []
        st = self._stmts(final)
        while self.done_stmts < len(st):
            out += self._run(st[self.done_stmts], self.done_stmts)
            self.done_stmts += 1
        return out

    def feed(self, delta):
        if self.finished:
            return []
        self.text += delta or ""
        return self._advance(False)

    def finish(self):
        if self.finished:
            return []
        out = self._advance(True)
        self.finished = True
        if self.done_stmts == 0 or not any(self.exec_ms):
            pass
        if not self.keep and self.channels:
            poses = [{"t": round(self.t + i / self.hz, 4), "head_yaw": 0.0} for i in range(1, 8)]
            out.append({"stmt": self.done_stmts, "t0": self.t, "t1": poses[-1]["t"], "poses": poses, "final": True})
            self.t = poses[-1]["t"]
        elif out:
            out[-1]["final"] = True
        if self.done_stmts == 0:
            self.errors.append({"stmt": None, "error": "no statements in the answer"})
        return out

    def report(self):
        return {"program": self.program, "n_statements": self.done_stmts, "duration_s": round(self.t, 3),
                "warnings": list(self.warnings), "errors": list(self.errors), "exec_ms": list(self.exec_ms),
                "keep_pose": self.keep, "channels": sorted(self.channels)}


def replay(character, program, targets=None, seed=0, hz=20):
    s = Stream(character, targets, seed, hz)
    poses = []
    for b in s.feed("```python\n" + program + "\n```\n") + s.finish():
        poses += b["poses"]
    return poses, s.report()
