"""Streaming execution of a semantic program: text deltas in -> dense rig-space pose batches out.

TIME COMMITMENT: the interpreter's rendering is causal (a frame at time t depends only on segments that started at or
before t), and every top-level statement starts at the sequence cursor (the end of the previous one).  So once
statement k is complete, every frame before its end time is final: feed() emits the poses with t < cursor_k and never
revises them; the next statement only adds poses after the last emitted one.  finish() adds the remaining statements,
the automatic hold + return-to-neutral tail (unless keep_pose) and emits the rest.
"""
from __future__ import annotations

import ast
import copy
import math
import re
import threading
import time

import numpy as np

from . import fastrender, interp, kin
from .rigmap import RigMap, driven_params, REST

FPS = interp.FPS
MAX_T = 12.0               # hard cap on motion length (s)
MAX_BODY = 10.9            # program statements are cut here; the tail (0.35 hold + 0.7 return) still fits in MAX_T
MAX_STMTS = 60
TAIL_HOLD, TAIL_RET = 0.35, 0.7
NAT_LIVE = dict(interp.NAT_DEFAULT, idle=False, blinks=False, smooth=False)
_CONT = re.compile(r"^(else|elif|except|finally)\b|^[\)\]\}]")

_RIGMAPS = {}
_RM_LOCK = threading.Lock()


def rigmap(character):
    with _RM_LOCK:
        r = _RIGMAPS.get(character)
        if r is None:
            r = _RIGMAPS[character] = RigMap(character)
        return r


def _fence_code(text, final):
    """-> (code text, closed) from the model's reply so far; None if the code block has not started."""
    i = text.find("```")
    if i < 0:
        if final:
            return text.strip("\n"), True
        s = text.lstrip()
        # a reply that starts straight with code (no fence): treat it as code once it clearly is not prose
        if len(s) >= 12 and re.match(r"^(with |for |[a-z_]+\()", s):
            return s, False
        return None, False
    j = text.find("\n", i)
    if j < 0:
        return None, False
    body = text[j + 1:]
    k = body.find("```")
    if k >= 0:
        return body[:k], True
    return body, final


class Stream:
    def __init__(self, character, targets=None, seed=0, hz=20, t_origin=0.0):
        self.character = character
        self.rm = rigmap(character)
        self.targets = dict(targets or {})
        self.seed = int(seed)
        self.hz = float(hz) if hz and hz > 0 else 20.0
        self.t_origin = float(t_origin)
        self.lock = threading.Lock()
        self.text = ""
        self.line = 0                    # next unconsumed code line
        self.B = interp.Builder(dict(NAT_LIVE))
        self.I = interp.Interp(self.B, strict=False)
        self.n_stmt = 0
        self.stmt_src = []
        self.errors = []
        self.exec_ms = []
        self.emitted_k = 0               # next output pose index (t = k / hz)
        self.finished = False
        self.channels = set()
        self.last_pose = None
        self.batches_out = 0
        self.code_closed = False
        self.extent = {}
        self.defer = False               # replay: render once at the end (same numbers: the renderer is causal)

    # ------------------------------------------------------------------ public
    def feed(self, delta):
        with self.lock:
            if self.finished or not delta:
                return []
            self.text += str(delta)
            return self._advance(final=False)

    def finish(self):
        with self.lock:
            if self.finished:
                return []
            out = self._advance(final=True)
            self.finished = True
            t0 = time.perf_counter()
            out += self._emit_tail()
            if out:
                self.exec_ms.append(round((time.perf_counter() - t0) * 1000, 2))
            if out:
                out[-1]["final"] = True
            else:
                out = [{"stmt": self.n_stmt, "t0": self._t(self.emitted_k), "t1": self._t(self.emitted_k),
                        "poses": [], "final": True}]
                if self.n_stmt == 0 and not self.errors:
                    self.errors.append("no valid statements in the program")
            return out

    def report(self):
        with self.lock:
            dur = self._end_time() if self.finished else self._cursor()
            return {"program": self._code_all(), "n_statements": self.n_stmt, "duration_s": round(dur, 3),
                    "warnings": list(self.B.warnings), "errors": list(self.errors), "exec_ms": list(self.exec_ms),
                    "keep_pose": bool(self.B.keep_end_pose), "channels": sorted(self.channels),
                    "extent": self._extent()}

    def _extent(self):
        """framing hint: how far the hands go (rig hand units) and whether the arms leave a head-and-shoulders
        close-up (out wide |x| > 0.55, or swinging between the lap and above the chest)."""
        e = {k: round(v, 3) for k, v in self.extent.items()}
        xs = [abs(e.get("hand_x_max", 0.0)), abs(e.get("hand_x_min", 0.0))]
        e["arms_leave_closeup"] = bool(max(xs) > 0.55)
        e["whole_body"] = any(k.startswith("root_") and v > 0.02 for k, v in e.items() if isinstance(v, float))
        return e

    # ------------------------------------------------------------------ statements
    def _code_all(self):
        c, _ = _fence_code(self.text, True)
        return interp.extract_code(self.text) if c is None else c

    def _advance(self, final):
        code, closed = _fence_code(self.text, final)
        if code is None:
            return []
        self.code_closed = closed
        lines = code.split("\n")
        n_full = len(lines) if closed else len(lines) - 1          # the last line may still be growing
        out = []
        while self.line < n_full:
            start = self.line
            # skip blank / comment lines
            if not lines[start].strip() or lines[start].lstrip().startswith("#"):
                self.line += 1
                continue
            bounds = [j for j in range(start + 1, n_full) if lines[j].strip() and lines[j][0] not in " \t#"
                      and not _CONT.match(lines[j])]
            if closed:
                bounds.append(n_full)
            if not bounds:
                # early completion (hub ask): a simple statement whose lines so far parse is complete now
                try:
                    tree = ast.parse("\n".join(lines[start:n_full]))
                except SyntaxError:
                    break
                if not tree.body or isinstance(tree.body[-1], (ast.With, ast.For, ast.If, ast.While)):
                    break
                bounds = [n_full]
            done = False
            for bi, j in enumerate(bounds):
                chunk = "\n".join(lines[start:j])
                try:
                    tree = ast.parse(chunk)
                except SyntaxError as e:
                    # a genuinely bad statement (the next chunk parses alone) or the end: skip it, record it
                    nxt = bounds[bi + 1] if bi + 1 < len(bounds) else None
                    alone_ok = False
                    if nxt is not None:
                        try:
                            ast.parse("\n".join(lines[j:nxt]))
                            alone_ok = True
                        except SyntaxError:
                            pass
                    if alone_ok or (closed and nxt is None):
                        self.errors.append(f"line {start + 1}: syntax error ({e.msg}); statement skipped")
                        self.line = j
                        done = True
                        break
                    continue
                for st in tree.body:
                    b = self._exec_stmt(st, chunk)
                    if b:
                        out.append(b)
                self.line = j
                done = True
                break
            if not done:
                break
        return out

    def _exec_stmt(self, st, src):
        t0 = time.perf_counter()
        B = self.B
        if self.n_stmt >= MAX_STMTS:
            self._err(f"more than {MAX_STMTS} statements: the rest is ignored")
            return None
        if self._cursor() >= MAX_BODY:
            self._err(f"program longer than {MAX_BODY:.1f} s: statement at line {getattr(st, 'lineno', '?')} ignored")
            return None
        snap = (len(B.segs), len(B.oscs), copy.deepcopy(B.stack), dict(B.last_arm), B.ended_with_return,
                B.keep_end_pose, set(B.expr_keys), set(B.explicit_keys), len(B.goals), len(self.I.errors),
                dict(self.I.env), self.I.steps)
        tok_t = interp.LIVE_TARGETS.set({k: tuple(v) for k, v in self.targets.items()})
        tok_w = interp.WARN_SINK.set(B.warnings)
        try:
            self.I.static_check(ast.Module(body=[st], type_ignores=[]))
            self.I.stmt(st, top=True)
        except Exception as e:   # noqa: BLE001 — a bad statement is skipped, never raised
            (ns, no, stack, la, ewr, kep, ek, xk, ng, ne, env, steps) = snap
            del B.segs[ns:]; del B.oscs[no:]; B.stack = stack; B.last_arm = la; B.ended_with_return = ewr
            B.keep_end_pose = kep; B.expr_keys = ek; B.explicit_keys = xk; del B.goals[ng:]
            self.I.env = env
            msg = str(e) if isinstance(e, interp.ProgramError) else f"line {getattr(st, 'lineno', '?')}: {type(e).__name__}: {e}"
            self._err(msg)
            return None
        finally:
            interp.LIVE_TARGETS.reset(tok_t)
            interp.WARN_SINK.reset(tok_w)
        for ln, msg in self.I.errors[snap[9]:]:
            self._err(msg)
        self.n_stmt += 1
        self.stmt_src.append(ast.get_source_segment(src, st) or "")
        # cut the statement's segments at MAX_BODY
        for s in B.segs[snap[0]:]:
            if s.t0 >= MAX_BODY:
                s.t0 = MAX_BODY - 1.0 / FPS
            s.t1 = min(s.t1, MAX_BODY)
        for o in B.oscs[snap[1]:]:
            o.t1 = min(o.t1, MAX_BODY)
        k1 = self._k_before(min(self._cursor(), MAX_BODY))
        b = None if self.defer else self._emit(k1)
        self.exec_ms.append(round((time.perf_counter() - t0) * 1000, 2))
        return b

    def _err(self, msg):
        if msg not in self.errors:
            self.errors.append(msg)

    # ------------------------------------------------------------------ time
    def _cursor(self):
        c = self.B.stack[0]
        return max(c["t"], c["end"])

    def _end_time(self):
        end = max([self._cursor()] + [s.t1 for s in self.B.segs] + [o.t1 for o in self.B.oscs] + [0.0])
        return min(end, MAX_T)

    def _t(self, k):
        return k / self.hz

    def _k_before(self, t):
        """number of output poses with time strictly < t"""
        return int(math.ceil(t * self.hz - 1e-9))

    # ------------------------------------------------------------------ rendering + emission
    def _render(self, k_end):
        """rig tracks for output poses 0..k_end-1 (the renderer is causal: earlier poses never change)."""
        T = max(1, int(k_end))
        return fastrender.render(self.B, self.character, T, self.hz), T

    def _emit(self, k1, final=False):
        k0 = self.emitted_k
        if k1 <= k0:
            return None
        rig, T = self._render(k1)
        spans = driven_params(self.B, self.character)
        ts = np.array([self._t(k) for k in range(k0, k1)])
        poses = [{"t": round(float(t) + self.t_origin, 4)} for t in ts]
        for p, sp in spans.items():
            if p not in rig:
                continue
            vals = rig[p][k0:k1]
            sp = sorted(sp)
            starts = np.array([a for a, _ in sp])
            ends = np.maximum.accumulate(np.array([b for _, b in sp]))
            rest = REST.get(p, 0.0)
            for q, t in enumerate(ts):
                j = int(np.searchsorted(starts, t + 1e-9, side="right")) - 1
                if j < 0:
                    continue                    # not driven yet
                last = ends[j]                  # causal: only spans that have started by now
                if t > last + 0.05 and abs(vals[q] - rest) < 2e-3:
                    continue                    # done with it and back at rest: the live layer takes it back
                poses[q][p] = round(float(vals[q]), 4)
                self.channels.add(p)
        for q in poses:                          # framing hint for the page (report.extent)
            for k, v in q.items():
                if k.startswith("hand_") and k[-2:] in ("_x", "_y", "_z"):
                    a = k[-1]
                    e = self.extent
                    e["hand_" + a + "_max"] = max(e.get("hand_" + a + "_max", -9.0), v)
                    e["hand_" + a + "_min"] = min(e.get("hand_" + a + "_min", 9.0), v)
                elif k in ("root_x", "root_z", "root_y"):
                    self.extent[k + "_abs_max"] = max(self.extent.get(k + "_abs_max", 0.0), abs(v))
        self.emitted_k = k1
        self.last_pose = poses[-1] if poses else self.last_pose
        return {"stmt": self.n_stmt, "t0": round(float(ts[0]) + self.t_origin, 4),
                "t1": round(float(ts[-1]) + self.t_origin, 4), "poses": poses, "final": final}

    def _emit_tail(self):
        B = self.B
        if self.n_stmt == 0:
            return []
        end = min(max([self._cursor()] + [s.t1 for s in B.segs] + [o.t1 for o in B.oscs]), MAX_BODY)
        if not B.ended_with_return and not B.keep_end_pose:
            t0 = max(end + TAIL_HOLD, self._t(self.emitted_k))
            for c in ["root", "rootp", "torso", "head", "gaze", "arm_l", "arm_r", "fing_l", "fing_r", "face",
                      "shrug_l", "shrug_r", "wrist_l", "wrist_r", "tongue"]:
                tg = {"neutral": True} if not c.startswith(("shrug", "wrist", "rootp", "tongue")) else (
                    {"amount": 0.0} if c == "tongue" else
                    {"amount": 0.0} if c.startswith("shrug") else {"wr": (0.0, 0.0, 0.0)} if c.startswith("wrist")
                    else {"pitch_add": 0.0})
                if c == "rootp":
                    continue
                if not any(s.chan == c for s in B.segs) and not any(o.chan in (c, c.replace("arm_", "hand_")) for o in B.oscs):
                    continue                # never driven: nothing to return
                B.segs.append(interp.Seg(c, t0, t0 + TAIL_RET, tg, B.ease(None), "neutral", "auto_return"))
            end = t0 + TAIL_RET
        k1 = self._k_before(min(end, MAX_T)) + 1
        b = self._emit(k1, final=True)
        return [b] if b else []


def replay(character, program, targets=None, seed=0, hz=20):
    """whole program at once -> (poses, report); same numbers as streaming it."""
    s = Stream(character, targets=targets, seed=seed, hz=hz)
    s.defer = True
    text = program if "```" in program else "```python\n" + program.strip("\n") + "\n```\n"
    bs = s.feed(text)
    bs += s.finish()
    poses = [p for b in bs for p in b["poses"]]
    return poses, s.report()
