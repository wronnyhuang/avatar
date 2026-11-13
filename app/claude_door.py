"""claude_door -- in-app streaming text-model door to Claude.

Two backends with one interface:
  * ClaudeDoor: shells to the `claude` CLI (its own login, no API key). A warm pool of pre-spawned
    `claude -p --input-format stream-json ...` processes; each serves ONE request (fresh context) and is replaced in
    the background. See docs/CLAUDE-DOOR.md.
  * ApiDoor: the Anthropic Messages API (ANTHROPIC_API_KEY, streamed), used when the CLI is not installed.
`door` is the process-wide instance: the CLI when it is found (AVATAR_CLAUDE_BIN or `claude` on PATH), else the API
when ANTHROPIC_API_KEY is set, else a ClaudeDoor whose requests fail with a clear "not found" error.

    from claude_door import door
    async for ev in door.stream(model="haiku", system=..., user=..., images=None, max_s=20, tag="board"):
        ev: {"delta": str} ... then {"done": True, "ok", "text", "ms_first", "ms_done", "error", "model", ...}
    ev = await door.complete(...)
    door.prewarm(model, system, n=2)
    await door.close()
"""
import asyncio
import base64
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time

DEFAULT_PER_PAIR = 2
MAX_TOTAL = 6
WARM_TTL_S = 240.0          # recycle idle warm processes (OAuth/cache freshness, memory)
_MODELS = {"haiku", "sonnet", "opus"}


def _claude_bin():
    b = os.environ.get("AVATAR_CLAUDE_BIN")
    if b:
        return b
    return shutil.which("claude") or "claude"


def cli_available():
    b = os.environ.get("AVATAR_CLAUDE_BIN")
    if b:
        return bool((os.path.isfile(b) and os.access(b, os.X_OK)) or shutil.which(b))
    return shutil.which("claude") is not None


def _flags(model, system, effort):
    f = ["-p", "--input-format", "stream-json", "--output-format", "stream-json",
         "--include-partial-messages", "--verbose", "--model", model,
         "--system-prompt", system, "--tools", "",
         "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
         "--no-session-persistence"]
    if effort:
        f += ["--effort", effort]
    return f


class _Slot:
    """One claude process, used for exactly one request."""

    def __init__(self, key, proc):
        self.key = key
        self.proc = proc
        self.born = time.monotonic()
        self.err = bytearray()
        self.err_task = asyncio.ensure_future(self._drain_err())

    async def _drain_err(self):
        try:
            while True:
                b = await self.proc.stderr.read(4096)
                if not b:
                    return
                if len(self.err) < 8192:
                    self.err += b
        except Exception:
            pass

    def alive(self):
        return self.proc.returncode is None

    def kill(self):
        p = self.proc
        if p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass
        asyncio.ensure_future(self._reap())

    async def _reap(self):
        try:
            await asyncio.wait_for(self.proc.wait(), 5)
        except Exception:
            pass
        self.err_task.cancel()


class ClaudeDoor:
    def __init__(self):
        self._pool = {}          # key -> [slot,...] idle warm
        self._want = {}          # key -> n warm to keep
        self._spawning = {}      # key -> count in flight
        self._cwd = None
        self._closed = False
        self._tasks = set()
        self.effort = os.environ.get("AVATAR_CLAUDE_EFFORT", "low") or None   # low: stops adaptive hidden thinking (+1-2.5 s first token)
        # thinking=False starts the processes with MAX_THINKING_TOKENS=0 — measured: --effort low still
        # opens a thinking block on ~1 in 4 planner answers (Haiku: first text 5-8 s instead of ~1.3 s); 0 removes it
        self.thinking = os.environ.get("AVATAR_CLAUDE_THINKING", "").strip().lower() not in ("0", "off", "false", "no")

    # ---- process management -------------------------------------------------
    def _workdir(self):
        if not self._cwd or not os.path.isdir(self._cwd):
            self._cwd = tempfile.mkdtemp(prefix="claude-door-")
        return self._cwd

    def _total(self):
        return sum(len(v) for v in self._pool.values()) + sum(self._spawning.values())

    async def _spawn(self, key):
        model, system = key
        env = dict(os.environ)
        env.setdefault("HOME", os.path.expanduser("~"))
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        env.pop("ANTHROPIC_API_KEY", None)   # the CLI's own login only
        if not self.thinking:
            env["MAX_THINKING_TOKENS"] = "0"
        if not cli_available():
            raise RuntimeError("the `claude` CLI was not found (install Claude Code, or set AVATAR_CLAUDE_BIN; "
                               "or set ANTHROPIC_API_KEY to use the Anthropic API instead)")
        proc = await asyncio.create_subprocess_exec(
            _claude_bin(), *_flags(model, system, self.effort),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=self._workdir(), env=env,
            limit=1 << 22, start_new_session=True)
        return _Slot(key, proc)

    def _bg(self, coro):
        t = asyncio.ensure_future(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _fill(self, key):
        want = self._want.get(key, 0)
        while not self._closed:
            have = len(self._pool.get(key, [])) + self._spawning.get(key, 0)
            if have >= want or self._total() >= MAX_TOTAL:
                return
            self._spawning[key] = self._spawning.get(key, 0) + 1
            try:
                s = await self._spawn(key)
            except Exception:
                return
            finally:
                self._spawning[key] -= 1
            self._pool.setdefault(key, []).append(s)

    def prewarm(self, model, system, n=DEFAULT_PER_PAIR):
        key = (model, system)
        self._want[key] = max(0, min(int(n), MAX_TOTAL))
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._bg(self._fill(key))
        if not getattr(self, "_janitor", None):
            self._janitor = asyncio.ensure_future(self._janitor_loop())

    async def _janitor_loop(self):
        while not self._closed:
            await asyncio.sleep(30)
            now = time.monotonic()
            for key, lst in list(self._pool.items()):
                keep = []
                for s in lst:
                    if not s.alive() or now - s.born > WARM_TTL_S:
                        s.kill()
                    else:
                        keep.append(s)
                self._pool[key] = keep
                if self._want.get(key):
                    self._bg(self._fill(key))

    async def _acquire(self, key):
        lst = self._pool.get(key, [])
        while lst:
            s = lst.pop(0)
            if s.alive():
                warm = True
                break
            s.kill()
        else:
            s, warm = None, False
        if s is None:
            s = await self._spawn(key)
        if self._want.get(key):
            self._bg(self._fill(key))
        return s, warm

    # ---- request ------------------------------------------------------------
    @staticmethod
    def _content(user, images):
        if not images:
            return user
        blocks = []
        for p in images:
            mt = mimetypes.guess_type(p)[0] or "image/png"
            with open(p, "rb") as fh:
                data = base64.b64encode(fh.read()).decode()
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
        blocks.append({"type": "text", "text": user})
        return blocks

    async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="door"):
        t0 = time.monotonic()
        ms = lambda: int((time.monotonic() - t0) * 1000)
        text, first, err, usage, cost = [], None, None, None, None
        slot = None
        warm = False
        try:
            if self._closed:
                raise RuntimeError("door closed")
            if model not in _MODELS and not model.startswith("claude-"):
                raise ValueError("bad model %r" % model)
            payload = json.dumps({"type": "user", "message": {"role": "user",
                                  "content": self._content(user, images)}}) + "\n"
            slot, warm = await self._acquire((model, system))
            slot.proc.stdin.write(payload.encode())
            await slot.proc.stdin.drain()
            deadline = t0 + max_s
            finished = False
            while not finished:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    err = "timeout after %.1fs" % max_s
                    break
                try:
                    line = await asyncio.wait_for(slot.proc.stdout.readline(), rem)
                except asyncio.TimeoutError:
                    err = "timeout after %.1fs" % max_s
                    break
                except ValueError:
                    err = "stream line too long"
                    break
                if not line:
                    await asyncio.sleep(0.05)
                    err = "claude exited rc=%s: %s" % (slot.proc.returncode, bytes(slot.err).decode("utf8", "replace").strip()[-600:])
                    break
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                ty = d.get("type")
                if ty == "stream_event":
                    e = d.get("event") or {}
                    if e.get("type") == "content_block_delta":
                        dl = e.get("delta") or {}
                        if dl.get("type") == "text_delta" and dl.get("text"):
                            if first is None:
                                first = ms()
                            text.append(dl["text"])
                            yield {"delta": dl["text"]}
                elif ty == "result":
                    usage = d.get("usage")
                    cost = d.get("total_cost_usd")
                    if d.get("is_error") or d.get("subtype", "success") != "success":
                        err = (d.get("result") or d.get("subtype") or "error")
                        if isinstance(err, list):
                            err = json.dumps(err)[:400]
                    if not text and isinstance(d.get("result"), str) and not err:
                        text.append(d["result"])
                        first = first if first is not None else ms()
                        yield {"delta": d["result"]}
                    finished = True
        except GeneratorExit:
            err = "cancelled"
            raise
        except Exception as ex:   # noqa
            err = "%s: %s" % (type(ex).__name__, ex)
        finally:
            if slot is not None:
                slot.kill()      # one request per process, always
        full = "".join(text)
        yield {"done": True, "ok": err is None, "text": full, "ms_first": first if first is not None else -1,
               "ms_done": ms(), "error": err, "model": model, "tag": tag, "warm": warm,
               "usage": usage, "cost_usd": cost}

    async def complete(self, **kw):
        last = None
        async for ev in self.stream(**kw):
            last = ev
        return last

    async def close(self):
        self._closed = True
        j = getattr(self, "_janitor", None)
        if j:
            j.cancel()
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        for lst in self._pool.values():
            for s in lst:
                s.kill()
        self._pool.clear()
        await asyncio.sleep(0.1)
        if self._cwd:
            shutil.rmtree(self._cwd, ignore_errors=True)
            self._cwd = None


class ApiDoor:
    """The same interface as ClaudeDoor over the Anthropic Messages API (ANTHROPIC_API_KEY): stream() yields
    {"delta"} ... then the same {"done": True, ...} event; prewarm() is a no-op (no processes to keep warm)."""
    URL = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"
    ALIASES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
    PRICE = {"claude-haiku-5-5": (0.10, 0.01, 0.50), "claude-sonnet-5-5": (2.0, 0.10, 10.0),
             "claude-opus-5-5": (4.0, 0.20, 20.0)}            # USD / M tokens: in, cache read, out

    def __init__(self, max_tokens=16000):
        self.effort = os.environ.get("AVATAR_CLAUDE_EFFORT", "low") or None
        self.thinking = os.environ.get("AVATAR_CLAUDE_THINKING", "").strip().lower() not in ("0", "off", "false", "no")
        self.max_tokens = max_tokens
        self._http = None

    def prewarm(self, model, system, n=DEFAULT_PER_PAIR):
        return None

    def _body(self, model, system, user, images):
        mid = self.ALIASES.get(model, model)
        body = {"model": mid, "max_tokens": self.max_tokens, "stream": True,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": ClaudeDoor._content(user, images)}]}
        if self.effort:
            body["output_config"] = {"effort": self.effort}
        if not self.thinking:            # how each model turns thinking off (Opus 5.5 cannot: low effort only)
            if "haiku" in mid:
                body["thinking"] = {"type": "disabled"}
            elif "sonnet" in mid and (self.effort or "high") in ("low", "medium", "high"):
                body["thinking"] = {"type": "between_tools"}
        return mid, body

    def _usd(self, mid, u):
        pi, pc, po = self.PRICE.get(mid, (4.0, 0.2, 20.0))
        u = u or {}
        return ((u.get("input_tokens", 0) or 0) * pi + 1.25 * (u.get("cache_creation_input_tokens", 0) or 0) * pi
                + (u.get("cache_read_input_tokens", 0) or 0) * pc + (u.get("output_tokens", 0) or 0) * po) / 1e6

    async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag="door"):
        import aiohttp
        t0 = time.monotonic()
        ms = lambda: int((time.monotonic() - t0) * 1000)       # noqa: E731
        text, first, err, usage = [], None, None, {}
        mid = self.ALIASES.get(model, model)
        try:
            key = os.environ.get("ANTHROPIC_API_KEY") or ""
            if not key:
                raise RuntimeError("no ANTHROPIC_API_KEY (and no `claude` CLI): set one of them to use this feature")
            mid, body = self._body(model, system, user, images)
            headers = {"x-api-key": key, "anthropic-version": self.VERSION, "content-type": "application/json"}
            if "sonnet" in mid or "opus" in mid:       # route a policy decline to the server's default fallback model
                headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
                body["fallbacks"] = "default"
            if self._http is None or self._http.closed:
                self._http = aiohttp.ClientSession()
            deadline = t0 + max_s
            async with self._http.post(self.URL, json=body, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=max_s)) as r:
                if r.status != 200:
                    raise RuntimeError("Anthropic API HTTP %d: %s" % (r.status, (await r.text())[:300]))
                buf = b""
                done = False
                while not done:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        err = "timeout after %.1fs" % max_s
                        break
                    try:
                        chunk = await asyncio.wait_for(r.content.readany(), rem)
                    except asyncio.TimeoutError:
                        err = "timeout after %.1fs" % max_s
                        break
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        try:
                            e = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        ty = e.get("type")
                        if ty == "content_block_delta":
                            dl = e.get("delta") or {}
                            if dl.get("type") == "text_delta" and dl.get("text"):
                                if first is None:
                                    first = ms()
                                text.append(dl["text"])
                                yield {"delta": dl["text"]}
                        elif ty == "message_start":
                            usage.update(((e.get("message") or {}).get("usage")) or {})
                        elif ty == "message_delta":
                            usage.update({k: v for k, v in (e.get("usage") or {}).items() if v is not None})
                            if (e.get("delta") or {}).get("stop_reason") == "refusal":
                                err = "refusal"
                        elif ty == "error":
                            err = "Anthropic API error: %s" % json.dumps(e.get("error"))[:300]
                            done = True
                        elif ty == "message_stop":
                            done = True
        except GeneratorExit:
            err = "cancelled"
            raise
        except Exception as ex:   # noqa
            err = "%s: %s" % (type(ex).__name__, ex)
        full = "".join(text)
        yield {"done": True, "ok": err is None, "text": full, "ms_first": first if first is not None else -1,
               "ms_done": ms(), "error": err, "model": model, "tag": tag, "warm": False,
               "usage": usage or None, "cost_usd": round(self._usd(mid, usage), 6) if usage else None}

    async def complete(self, **kw):
        last = None
        async for ev in self.stream(**kw):
            last = ev
        return last

    async def close(self):
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None


def make_door():
    """a new door: the CLI when it is found, else the API when ANTHROPIC_API_KEY is set, else the CLI door (whose
    requests then fail naming what is missing)."""
    if cli_available() or not os.environ.get("ANTHROPIC_API_KEY"):
        return ClaudeDoor()
    return ApiDoor()


door = make_door()

# ---------------------------------------------------------------------------
# bench CLI
# ---------------------------------------------------------------------------
_BENCH_SYS = ("You are the board brain of a classroom avatar. " + "Answer in plain prose, concisely. " * 1) + \
    ("Style notes: be warm, concrete, avoid lists. Use short sentences. " * 60)
_BENCH_Q = "Explain in about 110 words why the sky looks blue at noon and red at sunset."


def _pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else float("nan")


async def _bench(models, n, conc):
    rows = []
    async def one(m):
        return await door.complete(model=m, system=_BENCH_SYS, user=_BENCH_Q, max_s=40)
    for m in models:
        for mode in ("cold", "warm"):
            res = []
            if mode == "warm":
                door.prewarm(m, _BENCH_SYS, n=2)
            for i in range(n):
                if mode == "cold":
                    door._want.clear()
                else:
                    await asyncio.sleep(3.0 if i == 0 else 1.5)   # let replacement spawn settle
                res.append(await one(m))
            rows.append((m, mode, 1, res))
        door.prewarm(m, _BENCH_SYS, n=3)
        await asyncio.sleep(3)
        for r in range(max(3, n // 3)):
            rs = await asyncio.gather(*[one(m) for _ in range(conc)])
            rows.append((m, "warm x%d" % conc, conc, rs)) if r == 0 else rows[-1][3].extend(rs)
            await asyncio.sleep(3)
        door._want.clear()
    print("%-7s %-9s %3s %4s | first p50/p90 | done p50/p90 | cache_read(avg)" % ("model", "mode", "n", "ok"))
    for m, mode, c, res in rows:
        ok = [r for r in res if r["ok"]]
        f = [r["ms_first"] for r in ok]
        d = [r["ms_done"] for r in ok]
        cr = [(r.get("usage") or {}).get("cache_read_input_tokens", 0) for r in ok]
        print("%-7s %-9s %3d %4d | %5d/%5d  | %5d/%5d | %d" % (m, mode, len(res), len(ok), _pct(f, .5), _pct(f, .9),
              _pct(d, .5), _pct(d, .9), sum(cr) / max(1, len(cr))))
        for r in res:
            if not r["ok"]:
                print("   err:", r["error"])


def main(argv):
    if "--bench" in argv:
        models = ["haiku", "sonnet"]
        n = 15
        conc = 3
        if "--models" in argv:
            models = argv[argv.index("--models") + 1].split(",")
        if "--n" in argv:
            n = int(argv[argv.index("--n") + 1])

        async def go():
            try:
                await _bench(models, n, conc)
            finally:
                await door.close()
        asyncio.run(go())
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
