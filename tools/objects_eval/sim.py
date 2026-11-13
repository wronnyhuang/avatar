#!/usr/bin/env python3
"""objects-eval — step 1+2: one case through the product path without a human or a voice session.

Per spoken turn (setup turns, then the case's turns, one session from the home scene):
  * VOICE-MODEL STAND-IN: `claude -p --model sonnet` gets the candidate's OWN realtime session instructions + tool schemas
    (app/rtc.py realtime_session(..., character='muse', planner=True)) and the conversation so far (his words, its own
    earlier words and calls, tool results, the page's silent notes: action-log summaries + performer prompts) and answers
    with its next response {"say", "calls":[{name, arguments}]}. Cached by (instructions+tools hash, conversation).
    A response with calls is followed by another response (voice.js: a tool-only response gets a follow-up), up to
    MAX_ROUNDS per turn; the follow-up is asked after the body went idle (the notes it got by then are in the conversation).
  * --direct: the case's `direct` list (one list of {name, arguments} per turn, setup first) replaces the stand-in.
  * calls run through tools/objects_eval/engine.cjs (= page.html makeVoice onCall: voicetools -> performer -> object/engine);
    plan requests ({type:'plan'}) go to the candidate's REAL app/planner.py (Planner with its config defaults, plan cache
    <OBJECTS_EVAL_DIR>/cache/plans) and every kf message goes back into the engine (all at once, at the engine time the plan was
    asked: planner latency is not modelled). The engine is stepped until idle (cap MAX_IDLE_S) after every round.
Every frame is recorded; the case dumps rig.jsonl (+ .states.json) and case.json into the run's case dir.
"""
import asyncio, hashlib, json, os, re, subprocess, sys, tempfile, threading, time
from pathlib import Path

EVAL = Path(os.environ.get("OBJECTS_EVAL_DIR") or Path(__file__).resolve().parents[2] / "runs" / "objects_eval")   # workspace: candidate/, runs/, cache/
H = Path(__file__).resolve().parent                    # the harness (engine.cjs, render_keys.py, sync.sh) lives next to this file
CLAUDE_BIN = os.environ.get("AVATAR_CLAUDE_BIN") or "claude"
CLAUDE_CWD = Path(tempfile.gettempdir()) / "objects-eval-claude"   # outside the repo: no CLAUDE.md / project context in the stand-in
STANDIN_V = "standin-v1"
MAX_ROUNDS = 4
MAX_IDLE_S = 45.0
CLAUDE_SEM = threading.Semaphore(4)
_cost_lock = threading.Lock()
SRC = None
rtc = app_mod = kfplanner = None


def load_candidate(cand_dir):
    """import the candidate's app modules (one candidate per process)"""
    global SRC, rtc, app_mod, kfplanner
    SRC = Path(cand_dir) / "source"
    sys.path.insert(0, str(SRC / "app"))
    import rtc as _rtc, app as _app, planner as _pl  # noqa: E402
    rtc, app_mod, kfplanner = _rtc, _app, _pl


def session_spec():
    nm, what, vc = app_mod.CHARACTERS["muse"]
    s = rtc.realtime_session(nm, what, vc, "server:300", "", "parallel", character="muse", planner=True)
    tools = s.get("tools") or []
    h = hashlib.sha256((s.get("instructions", "") + json.dumps(tools, sort_keys=True)).encode()).hexdigest()[:16]
    return {"instructions": s.get("instructions", ""), "tools": tools, "hash": h}


# ------------------------------------------------------------------------------------------------- planner cost (estimate, this process)
_spent = 0.0


def spent_usd():
    """estimated planner cost so far in this process (list price, from the planner's usage reports)"""
    return _spent


def add_cost(usd, what):
    global _spent
    if not usd:
        return
    with _cost_lock:
        _spent += float(usd)


# ------------------------------------------------------------------------------------------------- claude -p
def claude(prompt, system=None, model="sonnet", image_tools=False, timeout=240, add_dirs=()):
    """one `claude -p` call through the subscription; -> (text, meta)"""
    CLAUDE_CWD.mkdir(parents=True, exist_ok=True)
    cmd = [CLAUDE_BIN, "-p", "--model", model, "--no-session-persistence", "--strict-mcp-config", "--output-format", "json"]
    if image_tools:
        cmd += ["--tools", "Read", "--permission-mode", "dontAsk", "--allowedTools", "Read"]
        for d in add_dirs:
            cmd += ["--add-dir", str(d)]
    else:
        cmd += ["--tools", ""]
    if system:
        cmd += ["--system-prompt", system]
    last = None
    for attempt in range(3):
        with CLAUDE_SEM:
            try:
                r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=str(CLAUDE_CWD))
            except subprocess.TimeoutExpired:
                last = "timeout"; continue
        try:
            j = json.loads(r.stdout)
            if not j.get("is_error") and j.get("result") is not None:
                return j["result"], {"cost_list_usd": j.get("total_cost_usd"), "ms": j.get("duration_ms")}
            last = str(j.get("result") or j)[:300]
        except ValueError:
            last = (r.stdout + r.stderr)[-300:]
        time.sleep(3 + 5 * attempt)
    raise RuntimeError(f"claude -p failed: {last}")


def last_json(text):
    """the last top-level {...} object in a text"""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.S)
    if m:
        s = m.group(1)
    depth, start, best = 0, None, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                best = s[start:i + 1]
    if best is None:
        raise ValueError("no JSON object in: " + text[:200])
    return json.loads(best)


STANDIN_APPENDIX = """

=== TEXT STAND-IN MODE (evaluation harness) ===
You are standing in for the realtime voice model of this app. Everything above is your real session instructions. Your
available function tools are these JSON schemas (exactly what the realtime session declares):
{tools}

The user message holds the conversation so far: what HE said, what YOU said, the function calls YOU made with their
results, and the app's silent system notes (never read aloud). Produce YOUR NEXT RESPONSE exactly as the realtime model
would at this point, following your instructions and the tool descriptions literally: the function calls you would make
(in order, with complete arguments matching the schemas) and the words you would speak. Output ONLY one JSON object:
{{"say": "<your spoken words, may be empty>", "calls": [{{"name": "<tool>", "arguments": {{...}}}}]}}
"calls" is [] when you would make none. No prose outside the JSON."""


def render_conv(conv):
    out = []
    for c in conv:
        r = c["role"]
        if r == "user":
            out.append(f"HIM (spoken): {c['text']}")
        elif r == "note":
            out.append(f"[silent system note] {c['text']}")
        elif r == "assistant":
            if c.get("say"):
                out.append(f"YOU (spoken): {c['say']}")
            for k in c.get("calls") or []:
                out.append(f"YOU called {k['name']}({json.dumps(k.get('arguments') or {})})")
        elif r == "tool":
            out.append(f"[function result for {c['name']}] {json.dumps(c['result'])}")
    return "\n".join(out)


def standin(spec, conv, cache_dir):
    body = render_conv(conv)
    key = hashlib.sha256((STANDIN_V + spec["hash"] + body).encode()).hexdigest()[:24]
    f = Path(cache_dir) / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text()) | {"cached": True}
    system = spec["instructions"] + STANDIN_APPENDIX.format(tools=json.dumps(spec["tools"]))
    prompt = "Conversation so far:\n" + body + "\n\nYour next response (JSON only):"
    err = None
    for _ in range(2):
        txt, meta = claude(prompt, system=system)
        try:
            j = last_json(txt)
            calls = [{"name": str(k.get("name")), "arguments": k.get("arguments") or {}} for k in (j.get("calls") or []) if isinstance(k, dict)]
            out = {"say": str(j.get("say") or ""), "calls": calls, "meta": meta}
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(out))
            return out | {"cached": False}
        except Exception as e:  # noqa: BLE001
            err = f"{e}: {txt[:200]}"
    return {"say": "", "calls": [], "error": err}


# ------------------------------------------------------------------------------------------------- engine.cjs
class Engine:
    def __init__(self):
        self.p = subprocess.Popen(["node", str(H / "engine.cjs"), str(SRC)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.pending = []

    def __call__(self, **q):
        self.p.stdin.write(json.dumps(q) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("engine.cjs died")
        r = json.loads(line)
        for c in r.pop("ctl", None) or []:
            self.pending.append(("ctl", c))
        for p in r.pop("prompts", None) or []:
            self.pending.append(("prompt", p))
        if "error" in r and q.get("op") != "tool":
            raise RuntimeError(f"engine {q.get('op')}: {r['error']}")
        return r

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.p.kill()


def planner_config():
    c = dict(kfplanner.DEFAULTS)
    d = app_mod.DEFAULTS.get("planner")
    if isinstance(d, dict):
        c.update(d)
    elif d is not None:
        c["enabled"] = bool(d)
    return c


class Case:
    def __init__(self, case, run_dir, direct=False, max_usd=1.5):
        self.c, self.dir, self.direct, self.max_usd = case, Path(run_dir) / case["id"], direct, max_usd
        self.dir.mkdir(parents=True, exist_ok=True)
        self.E = Engine()
        self.spec = session_spec()
        self.pcfg = planner_config()
        self.planner = kfplanner.Planner(self.pcfg, self._plog, cache_dir=str(EVAL / "cache/plans"))
        self.conv, self.plans, self.prompts, self.calls, self.plog, self.errors = [], [], [], [], [], []
        self.note_ver = None
        self.turn_k = -1
        self.plan_usd = 0.0

    def _plog(self, s):
        self.plog.append(str(s)[:300])

    # ---- side effects of every engine op
    def drain(self):
        while self.E.pending:
            kind, x = self.E.pending.pop(0)
            if kind == "ctl":
                if x.get("type") == "plan":
                    self.run_plan(x)
            else:
                self.prompts.append(dict(x, turn=self.turn_k))
                self.conv.append({"role": "note", "text": x.get("note", ""), "respond": x.get("respond")})

    def run_plan(self, c):
        pid = str(c.get("id") or "")
        desc = str(c.get("description") or "").strip()[:600]
        state = c.get("state") if isinstance(c.get("state"), dict) else None
        msgs, rec = [], {"id": pid, "description": desc[:200], "turn": self.turn_k, "t": self.E(op="now")["now"]}
        why = ("planner off" if not self.planner.enabled else "empty description" if not desc else
               "eval budget reached" if spent_usd() >= self.max_usd else "")
        if why:
            msgs.append(kfplanner.kf_message(pid, kfplanner.refusal(why)))
            rec.update(ok=False, error=why)
        else:
            async def go():
                import aiohttp
                async with aiohttp.ClientSession() as http:
                    kw = {}
                    import inspect
                    if "words" in inspect.signature(kfplanner.run_plan).parameters:
                        kw["words"] = str(c.get("words") or "").strip()[:400] or None
                    return await kfplanner.run_plan(self.planner, pid, c.get("character") or "muse", desc, state, msgs.append,
                                                    http=http, on_usd=self._usd, **kw)
            t0 = time.time()
            try:
                r = asyncio.run(go())
                rec.update({k: r.get(k) for k in ("ok", "source", "summary", "ms_first_pose", "ms_done", "usd", "error", "n_poses", "engine")})
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, error=f"harness: {e!r}"[:300])
                msgs.append(kfplanner.kf_message(pid, kfplanner.refusal(f"planner error: {e!r}"[:200])))
            rec["wall_s"] = round(time.time() - t0, 2)
        for m in msgs:
            r = self.E(op="kf", msg=m)
            if not r.get("ok"):
                rec["late_kf"] = rec.get("late_kf", 0) + 1
        rec["msgs"] = len(msgs)
        self.plans.append(rec)
        self.drain()

    def _usd(self, u):
        self.plan_usd += u
        add_cost(u, f"planner {self.pcfg.get('model')} {self.c['id']}")

    def settle(self, max_s=MAX_IDLE_S):
        t = 0.0
        while t < max_s:
            self.E(op="step", seconds=0.2); t += 0.2
            self.drain()
            if t >= 0.6 and self.E(op="isidle")["idle"] and not self.E.pending:
                break
        self.E(op="step", seconds=0.8); self.drain()
        return round(t, 1), self.E(op="isidle")["idle"]

    def alog_note(self):
        v = self.E(op="alogver")
        if v["v"] != self.note_ver:
            s = self.E(op="summary")["summary"]
            self.conv.append({"role": "note", "text": s[:1400], "alog": True})
            self.note_ver = v["v"]

    def turn(self, text, direct_calls=None):
        self.turn_k += 1
        k = self.turn_k
        n0 = self.E(op="now")
        rec = {"k": k, "text": text, "t0": n0["now"], "frame0": n0["frame"], "rounds": [], "said": []}
        self.alog_note()
        self.E(op="user", speech=True); self.E(op="step", seconds=1.5); self.E(op="user", stop=True); self.drain()
        self.E(op="user_text", text=text, final=True); self.drain()
        self.conv.append({"role": "user", "text": text})
        rounds = [{"say": "", "calls": direct_calls}] if direct_calls is not None else None
        for r_i in range(MAX_ROUNDS if rounds is None else 1):
            if rounds is None:
                if r_i:
                    self.alog_note()
                resp = standin(self.spec, self.conv, EVAL / "cache/standin")
            else:
                resp = rounds[0]
            if resp.get("error"):
                self.errors.append({"turn": k, "standin": resp["error"]})
            self.conv.append({"role": "assistant", "say": resp.get("say", ""), "calls": resp.get("calls") or []})
            if resp.get("say"):
                rec["said"].append(resp["say"])
            rr = {"say": resp.get("say", ""), "calls": [], "cached": resp.get("cached")}
            for ci, call in enumerate(resp.get("calls") or []):
                now = self.E(op="now")["now"]
                info = {"call_id": f"c{k}_{r_i}_{ci}", "start_ms": round(now * 1000), "done_ms": round(now * 1000)}
                tr = self.E(op="tool", name=call["name"], args=call.get("arguments") or {}, info=info)
                out = {kk: v for kk, v in tr.items() if not kk.startswith("_")}
                self.conv.append({"role": "tool", "name": call["name"], "result": out})
                cr = {"name": call["name"], "args": call.get("arguments") or {}, "t": now, "result": out,
                      "cmd": tr.get("_cmd"), "via": tr.get("_via"), "deferred": tr.get("_deferred")}
                rr["calls"].append(cr); self.calls.append(dict(cr, turn=k))
                self.drain()
            rr["settle_s"], rr["idle"] = self.settle()
            rec["rounds"].append(rr)
            if not resp.get("calls"):
                break
        n1 = self.E(op="now")
        rec["t1"], rec["frame1"] = n1["now"], n1["frame"]
        return rec

    def run(self):
        c = self.c
        t0 = time.time()
        self.E(op="reset", seed=4408, character="muse", planner=True)
        self.E(op="step", seconds=0.5)
        texts = list(c.get("setup") or []) + list(c.get("turns") or [])
        n_setup = len(c.get("setup") or [])
        direct = c.get("direct") if self.direct else None
        turns = []
        for i, text in enumerate(texts):
            dc = direct[i] if direct is not None and i < len(direct) else (None if direct is None else [])
            turns.append(dict(self.turn(text, dc), setup=i < n_setup))
        main0 = turns[n_setup]["frame0"] if n_setup < len(turns) else 0
        t_main = turns[n_setup]["t0"] if n_setup < len(turns) else 0.0
        self.alog_note()
        summary = self.E(op="summary")
        state = self.E(op="rawstate")["state"]
        events = self.E(op="events")["events"]
        cmds = self.E(op="cmdlog")["cmds"]
        rig = self.dir / "rig.jsonl"
        d = self.E(op="dump", path=str(rig), id=c["id"])
        dm = self.E(op="dump", path=str(self.dir / "main.jsonl"), id=c["id"], **{"from": main0})
        perf = self.E(op="perf")
        self.E.close()
        out = {"id": c["id"], "kind": c.get("kind"), "setup": c.get("setup") or [], "turns_text": c.get("turns"),
               "expect": c.get("expect"), "mode": "direct" if self.direct else "standin", "spec_hash": self.spec["hash"],
               "turns": turns, "main_frame0": main0, "main_t0": t_main, "fps": 30,
               "summary": summary["summary"], "brief": summary["brief"], "state": state, "events": events, "cmds": cmds,
               "plans": self.plans, "prompts": self.prompts, "conv": self.conv, "errors": self.errors + (perf.get("errors") or []),
               "dump": d, "dump_main": dm, "plan_usd": round(self.plan_usd, 6), "wall_s": round(time.time() - t0, 1)}
        (self.dir / "case.json").write_text(json.dumps(out, indent=1))
        return out


def run_case(case, run_dir, direct=False, max_usd=1.5):
    try:
        return Case(case, run_dir, direct, max_usd).run()
    except Exception as e:  # noqa: BLE001
        import traceback
        err = {"id": case["id"], "error": repr(e)[:400], "trace": traceback.format_exc()[-2000:]}
        d = Path(run_dir) / case["id"]; d.mkdir(parents=True, exist_ok=True)
        (d / "case.json").write_text(json.dumps(err, indent=1))
        return err
