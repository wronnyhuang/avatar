"""planner model switch (app/planner.py): presets / route resolution / env overrides, the per-route request bodies
and SSE parsing (synthetic byte streams, no network), the hedged fallback on a real asyncio clock with fake streams,
the cache model rule, kf message shapes, the quality fill, and the route-dispatch latency. No paid calls.
    python3 tests/planner_routes_test.py"""
import asyncio
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "app")
sys.path.insert(0, APP)
os.environ["AVATAR_HOME"] = tempfile.mkdtemp(prefix="avatar-routes-home-")   # never the live server's state/data
ENV_KEYS = ("AVATAR_PLANNER_MODEL", "AVATAR_PLANNER_FALLBACK", "AVATAR_PLANNER_FALLBACK_AFTER_S")
for _k in ENV_KEYS:
    os.environ.pop(_k, None)
import planner  # noqa: E402

# model "auto" depends on the machine (claude CLI on PATH, which keys are set). These tests pin the auto choice to the
# OpenRouter pair so they are deterministic; test_auto_order below checks the real selection logic.
_real_auto_primary = planner.auto_primary
planner.auto_primary = lambda: "gem35lite"

DUMMY_KEYS = {"OPENAI_API_KEY": "test-openai", "OPENROUTER_API_KEY": "test-or", "GEMINI_API_KEY": "test-gem",
              "META_API_KEY": "test-meta"}     # the fake http never sends anything anywhere
NOLIB = {"library": False, "cache": False}


def run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------------------------------------------ resolution
class Resolve(unittest.TestCase):
    def setUp(self):
        for k in ENV_KEYS:
            os.environ.pop(k, None)

    tearDown = setUp

    def test_presets_reproduce_bench(self):
        R = planner.resolve
        want = {"sol": ("gpt-6.1-sol", "openai", "low"), "quality": ("gpt-6.1-sol", "openai", "low"),
                "gpt54mini": ("gpt-5.4-mini", "openai", "none"),
                "gem38flash": ("google/gemini-3.8-flash", "openrouter", "low"),
                "gem35lite": ("google/gemini-3.5-flash-lite", "openrouter", None),
                "gem35lite_gemini": ("gemini-3.5-flash-lite", "gemini", "minimal"),
                "glm53": ("z-ai/glm-5.3", "openrouter_zdr", "minimal"),
                "dsv41flash": ("deepseek/deepseek-v4.1-flash", "openrouter_zdr", "none"),
                "dsv4pro": ("deepseek/deepseek-v4-pro", "openrouter_zdr", "none"),
                "musespark11": ("muse-spark-1.1", "meta", "minimal"),
                "haiku55or": ("anthropic/claude-haiku-5.5", "openrouter", None),
                "haiku55api": ("claude-haiku-5-5", "anthropic", "none"),
                "sonnet55api": ("claude-sonnet-5-5", "anthropic", "low")}
        door = {k for k, p in planner.PRESETS.items() if p["route"] == "claude_door"}   # tests/planner_door_test.py
        self.assertEqual(set(want), set(planner.PRESETS) - door)
        for k, (m, r, e) in want.items():
            self.assertEqual(R({"model": k}), {"model": m, "route": r, "effort": e}, k)
        # every preset matches what the planner bench recorded for that tag (its stream.json settings), when the recordings exist
        out = os.path.join(os.environ.get("PLANNER_BENCH", ""), "out")
        if os.path.isdir(out):
            for k, (m, r, e) in want.items():
                d = os.path.join(out, "stream_kf-" + k)
                if not os.path.isdir(d):
                    continue
                with open(os.path.join(d, sorted(os.listdir(d))[0], "stream.json")) as fh:
                    s = json.load(fh)
                a = s["attempts"][-1]
                self.assertEqual((a["model"], a["route"], a["effort"]), (m, r, e), k)

    def test_auto_order(self):
        keys = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
        clean = {k: v for k, v in os.environ.items() if k not in keys}
        with mock.patch.object(planner, "claude_cli", lambda: ""):
            for env, want in [({}, "gem35lite"), ({"OPENAI_API_KEY": "x"}, "gpt54mini"),
                              ({"GEMINI_API_KEY": "x"}, "gem35lite_gemini"),
                              ({"OPENAI_API_KEY": "x", "OPENROUTER_API_KEY": "x"}, "gem35lite"),
                              ({"OPENROUTER_API_KEY": "x", "ANTHROPIC_API_KEY": "x"}, "haiku55api")]:
                with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
                    self.assertEqual(_real_auto_primary(), want, env)
        with mock.patch.object(planner, "claude_cli", lambda: "/usr/bin/claude"):
            self.assertEqual(_real_auto_primary(), "haiku55door")
        with mock.patch.dict(os.environ, {**clean, "OPENAI_API_KEY": "x"}, clear=True):
            self.assertEqual(planner.auto_fallback("haiku55door"), "gpt54mini")
            self.assertEqual(planner.auto_fallback("gem35lite"), "dsv41flash")

    def test_defaults_and_backcompat(self):
        self.assertEqual(planner.resolve(True), {"model": "google/gemini-3.5-flash-lite", "route": "openrouter",
                                                 "effort": None})
        self.assertEqual(planner.resolve_fallback(True), {"model": "deepseek/deepseek-v4.1-flash",
                                                          "route": "openrouter_zdr", "effort": "none", "after_s": 2.0})
        self.assertEqual(planner.resolve(None), planner.resolve(True))
        # the original Sol config dict keeps planning with Sol / effort low
        self.assertEqual(planner.resolve({"model": "gpt-6.1-sol", "effort": "low"}),
                         {"model": "gpt-6.1-sol", "route": "openai", "effort": "low"})
        self.assertIsNone(planner.resolve_fallback({"model": "sol", "fallback": None}))
        pl = planner.Planner({"model": "gpt-6.1-sol", "effort": "low", "timeout_s": 20}, lambda m: None)
        self.assertEqual(planner.resolve(pl.cfg), {"model": "gpt-6.1-sol", "route": "openai", "effort": "low"})
        self.assertEqual(pl.cfg["timeout_s"], 20)

    def test_raw_ids_and_routes(self):
        R = planner.resolve
        self.assertEqual(R({"model": "gpt-6.2-nano"}), {"model": "gpt-6.2-nano", "route": "openai", "effort": None})
        self.assertEqual(R({"model": "qwen/qwen-4"}), {"model": "qwen/qwen-4", "route": "openrouter", "effort": None})
        self.assertEqual(R({"model": "gemini-4-flash"})["route"], "gemini")
        self.assertEqual(R({"model": "muse-spark-2"})["route"], "meta")
        # GLM / DeepSeek always zero-retention (design rule), even when the config says plain openrouter
        self.assertEqual(R({"model": "deepseek/deepseek-v5", "route": "openrouter"})["route"], "openrouter_zdr")
        self.assertEqual(R({"model": "z-ai/glm-6"})["route"], "openrouter_zdr")
        self.assertEqual(R({"model": "qwen/qwen-4", "zdr": True})["route"], "openrouter_zdr")
        # a raw id one preset uses inherits that preset's route / effort; explicit keys win
        self.assertEqual(R({"model": "gpt-5.4-mini"})["effort"], "none")
        self.assertEqual(R({"model": "gpt54mini", "effort": "low"})["effort"], "low")
        self.assertIsNone(R({"model": "sol", "effort": None})["effort"])
        # the same model named for its route
        self.assertEqual(R({"model": "gem35lite", "route": "gemini"})["model"], "gemini-3.5-flash-lite")
        self.assertEqual(R({"model": "gemini-3.5-flash-lite", "route": "openrouter"})["model"],
                         "google/gemini-3.5-flash-lite")
        self.assertEqual(R({"model": "gpt-5.4-mini", "route": "openrouter"})["model"], "openai/gpt-5.4-mini")
        self.assertEqual(R({"model": "sol", "route": "bogus"})["route"], "openai")      # unknown route = auto

    def test_fallback_forms_and_env(self):
        F = planner.resolve_fallback
        self.assertEqual(F({"fallback": "gpt54mini"}), {"model": "gpt-5.4-mini", "route": "openai", "effort": "none",
                                                        "after_s": planner.FALLBACK_AFTER_S})
        self.assertEqual(F({"fallback": {"model": "glm53", "after_s": 1.25}})["after_s"], 1.25)
        self.assertEqual(F({"fallback": {"model": "x-ai/grok-5", "effort": "low"}})["effort"], "low")
        with mock.patch.dict(os.environ, {"AVATAR_PLANNER_MODEL": "sol", "AVATAR_PLANNER_FALLBACK": "gpt54mini",
                                          "AVATAR_PLANNER_FALLBACK_AFTER_S": "0.75"}):
            self.assertEqual(planner.resolve({"model": "glm53", "effort": "high"}),
                             {"model": "gpt-6.1-sol", "route": "openai", "effort": "low"})   # env: the preset as is
            self.assertEqual(F({"fallback": {"model": "glm53", "after_s": 5}}),
                             {"model": "gpt-5.4-mini", "route": "openai", "effort": "none", "after_s": 0.75})
            pl = planner.Planner(None, lambda m: None)
            self.assertEqual((pl.describe()["model"], pl.describe()["fallback"]["model"]), ("gpt-6.1-sol", "gpt-5.4-mini"))
        with mock.patch.dict(os.environ, {"AVATAR_PLANNER_MODEL": "z-ai/glm-5.3", "AVATAR_PLANNER_FALLBACK": "none"}):
            self.assertEqual(planner.resolve(True), {"model": "z-ai/glm-5.3", "route": "openrouter_zdr",
                                                     "effort": "minimal"})
            self.assertIsNone(F(True))

    def test_describe_and_stats(self):
        pl = planner.Planner({"model": "glm53", "fallback": {"model": "gpt54mini", "after_s": 1.5}}, lambda m: None)
        d = pl.describe()
        self.assertEqual((d["model"], d["route"], d["effort"], d["preset"]), ("z-ai/glm-5.3", "openrouter_zdr", "minimal",
                                                                              "glm53"))
        self.assertEqual(d["fallback"], {"model": "gpt-5.4-mini", "route": "openai", "effort": "none", "after_s": 1.5})
        self.assertEqual(d["quality"], {"model": "gpt-6.1-sol", "route": "openai", "effort": "low"})
        self.assertEqual(d["cache_models"], ["z-ai/glm-5.3", "gpt-5.4-mini", "gpt-6.1-sol"])
        self.assertEqual((pl.stats["model"], pl.stats["route"], pl.stats["fallback"]["model"]),
                         ("z-ai/glm-5.3", "openrouter_zdr", "gpt-5.4-mini"))
        json.dumps({"enabled": pl.enabled, **pl.stats})              # /api overview spreads stats into JSON


# ------------------------------------------------------------------------------------------------------ requests + SSE
def chunked(b: bytes, seed: int):
    """the same bytes cut at random places (mid-line, mid-UTF-8, mid-CRLF)"""
    rnd, out, i = random.Random(seed), [], 0
    while i < len(b):
        n = rnd.randint(1, 23)
        out.append(b[i:i + n])
        i += n
    return out


ANSWER = '{"summary":"hop – wave","poses":[{"t":0.3,"smile":0.5},{"t":0.8,"head_yaw":0.2},{"t":1.4,"smile":0}]}'
PIECES = [ANSWER[:20], ANSWER[20:45], ANSWER[45:70], ANSWER[70:]]


def sse(objs, crlf=False, extra=b""):
    nl = b"\r\n" if crlf else b"\n"
    out = extra
    for o in objs:
        out += b"data: " + (o if isinstance(o, bytes) else json.dumps(o, ensure_ascii=False).encode()) + nl + nl
    return out


def stream_openai():
    return sse([{"type": "response.created", "response": {"id": "r"}}]
               + [{"type": "response.output_text.delta", "delta": p} for p in PIECES]
               + [{"type": "response.completed", "response": {"usage": {"input_tokens": 3000, "input_tokens_details":
                                                                         {"cached_tokens": 2000}, "output_tokens": 400}}}],
               extra=b"event: response.created\n")


def stream_gemini():
    ev = [{"candidates": [{"content": {"role": "model", "parts": [{"text": "let me think", "thought": True},
                                                                   {"text": PIECES[0]}]}}]}]
    ev += [{"candidates": [{"content": {"role": "model", "parts": [{"text": p}]}}]} for p in PIECES[1:]]
    ev[-1]["usageMetadata"] = {"promptTokenCount": 3000, "candidatesTokenCount": 300, "thoughtsTokenCount": 100}
    ev[-1]["candidates"][0]["finishReason"] = "STOP"
    return sse(ev, crlf=True)


def stream_chat(cost=True):
    ev = [{"choices": [{"delta": {"role": "assistant", "content": ""}}]}]
    ev += [{"choices": [{"delta": {"content": p}}]} for p in PIECES]
    ev += [{"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    u = {"prompt_tokens": 3000, "prompt_tokens_details": {"cached_tokens": 1000}, "completion_tokens": 400}
    if cost:
        u["cost"] = 0.00123
    ev += [{"choices": [], "usage": u}]
    return sse(ev + [b"[DONE]"], extra=b": OPENROUTER PROCESSING\n\n" if cost else b"")


STREAMS = {"openai": stream_openai, "gemini": stream_gemini, "openrouter": stream_chat, "openrouter_zdr": stream_chat,
           "meta": lambda: stream_chat(cost=False)}


class FakeContent:
    def __init__(self, chunks, gap=0.0):
        self.chunks, self.gap = list(chunks), gap

    async def readany(self):
        if self.gap:
            await asyncio.sleep(self.gap)
        return self.chunks.pop(0) if self.chunks else b""


class FakeResp:
    def __init__(self, status=200, chunks=(), text="", gap=0.0):
        self.status, self.content, self._text = status, FakeContent(chunks, gap), text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def text(self):
        return self._text


class FakeHTTP:
    def __init__(self, make):
        self.make, self.calls = make, []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"t": time.perf_counter(), "url": url, "json": json, "headers": headers})
        return self.make(url, json)


class Requests(unittest.TestCase):
    def body(self, preset, **kw):
        sp = planner._spec({"model": preset, **kw})
        return planner.build_request(sp, "SYS", "USER", 12000)

    def test_bodies(self):
        url, b = self.body("sol")
        self.assertEqual(url, "https://api.openai.com/v1/responses")
        self.assertEqual(b, {"model": "gpt-6.1-sol", "instructions": "SYS", "input": "USER", "stream": True,
                             "text": {"format": {"type": "json_object"}}, "max_output_tokens": 12000,
                             "reasoning": {"effort": "low"}})
        self.assertEqual(self.body("gpt54mini")[1]["reasoning"], {"effort": "none"})      # sent 'none' as is
        self.assertNotIn("reasoning", self.body("sol", effort=None)[1])
        url, b = self.body("gem35lite_gemini")
        self.assertTrue(url.endswith("/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse"))
        self.assertNotIn("key=", url)                                                   # the key rides a header
        self.assertEqual(b["generationConfig"], {"responseMimeType": "application/json", "maxOutputTokens": 12000,
                                                 "thinkingConfig": {"thinkingLevel": "minimal"}})
        self.assertEqual(b["systemInstruction"], {"parts": [{"text": "SYS"}]})
        url, b = self.body("gem35lite")
        self.assertEqual(url, planner.OPENROUTER_URL)
        self.assertEqual(b, {"model": "google/gemini-3.5-flash-lite", "stream": True,
                             "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
                             "response_format": {"type": "json_object"}, "max_tokens": 12000,
                             "usage": {"include": True}})                               # no reasoning field
        b = self.body("dsv41flash")[1]
        self.assertEqual((b["reasoning"], b["provider"]), ({"effort": "none"}, {"zdr": True, "data_collection": "deny"}))
        self.assertEqual(self.body("glm53")[1]["provider"], {"zdr": True, "data_collection": "deny"})
        self.assertNotIn("provider", self.body("gem38flash")[1])
        url, b = self.body("musespark11")
        self.assertEqual(url, "https://api.meta.ai/v1/chat/completions")
        self.assertEqual((b["reasoning_effort"], b["stream_options"]), ("minimal", {"include_usage": True}))
        self.assertNotIn("usage", b)
        self.assertNotIn("reasoning", b)


class SSE(unittest.TestCase):
    def test_parse_each_route_any_chunking(self):
        for route, mk in STREAMS.items():
            raw = mk()
            for seed in range(25):
                texts, info, fin = planner.parse_sse(route, chunked(raw, seed))
                self.assertEqual("".join(texts), ANSWER, (route, seed))
                self.assertTrue(info.get("usage"), route)
                self.assertTrue(fin or route == "gemini", route)       # gemini ends at EOF, the others by event
        _, info, _ = planner.parse_sse("openrouter", [stream_chat()])
        self.assertEqual(planner.usd_of("google/gemini-3.5-flash-lite", info["usage"], "", "", ""), (0.00123, False))

    def test_errors(self):
        E = planner.PlanError
        with self.assertRaises(E) as c:
            planner.parse_sse("openrouter", [sse([{"choices": [{"delta": {"content": "{"}}]},
                                                  {"error": {"code": 502, "message": "upstream"}}])])
        self.assertTrue(c.exception.transient)
        with self.assertRaises(E) as c:
            planner.parse_sse("meta", [sse([{"error": {"code": 400, "message": "bad"}}])])
        self.assertFalse(c.exception.transient)
        with self.assertRaises(E) as c:
            planner.parse_sse("gemini", [sse([{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}])])
        self.assertTrue(c.exception.transient)
        with self.assertRaises(E) as c:
            planner.parse_sse("openai", [sse([{"type": "error", "error": {"type": "server_error"}}])])
        self.assertTrue(c.exception.transient)
        with self.assertRaises(E) as c:
            planner.parse_sse("openai", [sse([{"type": "response.incomplete", "response": {"usage": {"input_tokens": 1}}}])])
        self.assertFalse(c.exception.transient)
        # junk / comments / unknown events are skipped
        t, _, _ = planner.parse_sse("openrouter", [b": keepalive\n\ndata: {not json\n\nevent: x\n" + stream_chat()])
        self.assertEqual("".join(t), ANSWER)

    def test_route_stream_through_fake_http(self):
        """Planner._route_stream end to end per route: URL, auth header, body, deltas, usage — no network"""
        for route, mk in STREAMS.items():
            preset = {"openai": "sol", "gemini": "gem35lite_gemini", "openrouter": "gem35lite",
                      "openrouter_zdr": "dsv41flash", "meta": "musespark11"}[route]
            pl = planner.Planner({"model": preset, "fallback": None}, lambda m: None)
            http = FakeHTTP(lambda url, body, raw=mk(): FakeResp(200, chunked(raw, 7)))
            info = {"spec": pl.primary()}

            async def go():
                return [d async for d in pl._route_stream(http, "SYS", "USER", time.monotonic() + 5, info)]
            with mock.patch.dict(os.environ, DUMMY_KEYS):
                got = run(go())
            self.assertEqual("".join(got), ANSWER, route)
            self.assertTrue(info["usage"], route)
            h = http.calls[0]["headers"]
            if route == "gemini":
                self.assertEqual(h.get("x-goog-api-key"), DUMMY_KEYS["GEMINI_API_KEY"])
                self.assertNotIn("Authorization", h)
            else:
                env = planner.KEY_ENV[route]
                self.assertEqual(h["Authorization"], "Bearer " + DUMMY_KEYS[env])
            self.assertEqual(http.calls[0]["json"]["model"] if route != "gemini" else http.calls[0]["url"].split("/")[-1],
                             planner.resolve({"model": preset})["model"] if route != "gemini"
                             else "gemini-3.5-flash-lite:streamGenerateContent?alt=sse")

    def test_route_stream_http_errors_and_missing_key(self):
        pl = planner.Planner({"model": "gem35lite", "fallback": None}, lambda m: None)

        async def go(http):
            return [d async for d in pl._route_stream(http, "S", "U", time.monotonic() + 5, {"spec": pl.primary()})]
        for status, transient in ((429, True), (503, True), (400, False), (401, False)):
            http = FakeHTTP(lambda u, b, s=status: FakeResp(s, text='{"error":"Incorrect API key sk-abcdef123456"}'))
            with mock.patch.dict(os.environ, DUMMY_KEYS):
                with self.assertRaises(planner.PlanError) as c:
                    run(go(http))
            self.assertEqual((c.exception.transient, str(c.exception)[:8]), (transient, f"HTTP {status}"))
            self.assertNotIn("sk-abcdef", str(c.exception))                         # key-looking text redacted
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(planner.PlanError) as c:
                run(go(FakeHTTP(lambda u, b: FakeResp(200, [stream_chat()]))))
        self.assertEqual(str(c.exception), "no OPENROUTER_API_KEY")


class Cost(unittest.TestCase):
    def test_usd_shapes(self):
        U = planner.usd_of
        self.assertAlmostEqual(U("gpt-6.1-sol", {"input_tokens": 3101, "input_tokens_details": {"cached_tokens": 3079},
                                                 "output_tokens": 1090}, "", "", "")[0],
                               ((3101 - 3079) * 2.0 + 3079 * 0.10 + 1090 * 10.0) / 1e6)
        self.assertAlmostEqual(U("z-ai/glm-5.3", {"prompt_tokens": 3000, "completion_tokens": 400,
                                                  "prompt_tokens_details": {"cached_tokens": 1000}}, "", "", "")[0],
                               (2000 * 0.18 + 1000 * 0.18 + 400 * 1.85) / 1e6)
        self.assertEqual(U("z-ai/glm-5.3", {"prompt_tokens": 3000, "completion_tokens": 400, "cost": 0.0005}, "", "", ""),
                         (0.0005, False))                                            # OpenRouter's cost wins
        self.assertAlmostEqual(U("gemini-3.5-flash-lite", {"promptTokenCount": 3000, "candidatesTokenCount": 300,
                                                           "thoughtsTokenCount": 100}, "", "", "")[0],
                               (3000 * 0.1 + 400 * 0.4) / 1e6)
        self.assertEqual(U("muse-spark-1.1", {"prompt_tokens": 3000, "completion_tokens": 400}, "", "", "")[0], 0.0)
        u, est = U("deepseek/deepseek-v4.1-flash", None, "s" * 3300, "", "")          # cancelled before any text
        self.assertTrue(est and u > 0)
        self.assertEqual(planner.price_of("some-new-gpt"), (2.0, 0.10, 10.0))
        self.assertEqual(planner.price_of("acme/new-model"), (1.0, 1.0, 5.0))


# ------------------------------------------------------------------------------------------------------ hedged fallback
def pose_text(tag: float, n: int = 3):
    """a model's answer as pieces: the header, then one pose each (smile = tag marks the model), then the close"""
    return (['{"summary":"m%s","poses":[' % tag]
            + ['{"t":%s,"smile":%s,"head_yaw":%s},' % (0.3 + 0.5 * i, tag, i / 10) for i in range(n - 1)]
            + ['{"t":%s,"smile":%s}]}' % (0.3 + 0.5 * (n - 1), tag)])


def scripted(by_model: dict, calls: list | None = None):
    """stream_fn: per model id a list of steps (delay_s, text | Exception) — the fake vendor"""
    async def fn(http, system, user, deadline, info):
        m = info["spec"]["model"]
        if calls is not None:
            calls.append((m, time.monotonic()))
        for delay, x in by_model[m]:
            if delay:
                await asyncio.sleep(delay)
            if isinstance(x, BaseException):
                raise x
            yield x
        info["usage"] = {"prompt_tokens": 3000, "completion_tokens": 300}
    return fn


def steps(tag, first_at, every=0.02, n=3):
    t = pose_text(tag, n)
    return [(first_at, t[0] + t[1])] + [(every, x) for x in t[2:]]


PRIM, FB = "gpt-5.4-mini", "z-ai/glm-5.3"          # gpt54mini primary, glm53 fallback in these tests


def hedged(after_s, **kw):
    return {**NOLIB, "model": "gpt54mini", "fallback": {"model": "glm53", "after_s": after_s}, **kw}


async def collect(pl, desc="do something unusual", usd=None, character="muse"):
    evs = []
    async for ev in pl.plan(character, desc, None, http=object(), on_usd=(usd.append if usd is not None else None)):
        evs.append(ev)
    return evs


def smiles(evs):
    return {p["smile"] for e in evs if not e["done"] for p in (planner.wire_pose(x) for x in e["poses"]) if "smile" in p}


class Fallback(unittest.TestCase):
    def mk(self, cfg, script, calls=None):
        pl = planner.Planner(cfg, lambda m: None)
        pl.stream_fn = scripted(script, calls)
        return pl

    def test_primary_slow_fallback_wins(self):
        usd = []
        pl = self.mk(hedged(0.15), {PRIM: steps(0.1, 0.6), FB: steps(0.9, 0.05)})
        t0 = time.monotonic()
        evs = run(collect(pl, usd=usd))
        d = evs[-1]
        self.assertTrue(d["ok"] and d["fallback_used"] and d["fallback_started"])
        self.assertEqual((d["model"], d["route"]), (FB, "openrouter_zdr"))
        self.assertEqual(smiles(evs), {0.9})                                       # every pose from the fallback
        a0, a1 = d["attempts"]
        self.assertEqual((a0["role"], a0["model"], a0["cancelled"], a1["role"], a1["cancelled"]),
                         ("primary", PRIM, True, "fallback", False))
        self.assertTrue(140 <= a1["ms_start"] <= 260, a1["ms_start"])                # started at after_s
        self.assertTrue(180 <= d["ms_first_pose"] <= 330, d["ms_first_pose"])       # after_s + its first pose
        self.assertTrue(a0["usd"] > 0 and a0["estimated"])                          # the loser is billed (estimate)
        self.assertEqual(len(usd), 2)                                               # on_usd per attempt
        self.assertAlmostEqual(sum(usd), d["usd"], 6)
        self.assertLess(time.monotonic() - t0, 0.55)                                # never waited for the slow primary
        self.assertEqual((pl.stats["fallback_started"], pl.stats["fallback_used"]), (1, 1))

    def test_primary_pose_just_before_after_s(self):
        calls = []
        pl = self.mk(hedged(0.3), {PRIM: steps(0.1, 0.22), FB: steps(0.9, 0.01)}, calls)
        evs = run(collect(pl))
        d = evs[-1]
        self.assertTrue(d["ok"] and not d["fallback_used"] and not d["fallback_started"])
        self.assertEqual([m for m, _ in calls], [PRIM])                            # the fallback never started
        self.assertEqual((len(d["attempts"]), d["model"], smiles(evs)), (1, PRIM, {0.1}))

    def test_primary_error_starts_fallback_at_once(self):
        pl = self.mk(hedged(5.0), {PRIM: [(0.02, planner.PlanError("HTTP 503: busy", transient=True))],
                                   FB: steps(0.9, 0.05)})
        d = run(collect(pl))[-1]
        self.assertTrue(d["ok"] and d["fallback_used"])
        self.assertLess(d["attempts"][1]["ms_start"], 60)                          # not after_s later
        self.assertEqual(d["attempts"][0]["error"], "HTTP 503: busy")
        self.assertEqual([a["role"] for a in d["attempts"]], ["primary", "fallback"])   # no same-model retry

    def test_primary_no_poses_starts_fallback(self):
        pl = self.mk(hedged(5.0), {PRIM: [(0.02, '{"summary":"none"}')], FB: steps(0.9, 0.02)})
        d = run(collect(pl))[-1]
        self.assertTrue(d["ok"] and d["fallback_used"])

    def test_both_fail(self):
        usd = []
        pl = self.mk(hedged(0.05), {PRIM: [(0.15, planner.PlanError("HTTP 500: down", transient=True))],
                                    FB: [(0.02, '{"poses":['), (0.02, planner.PlanError("HTTP 400: bad model"))]})
        d = run(collect(pl, usd=usd))[-1]
        self.assertFalse(d["ok"])
        self.assertIn(f"primary {PRIM}: HTTP 500: down", d["error"])
        self.assertIn(f"fallback {FB}: HTTP 400: bad model", d["error"])
        self.assertEqual((d["all_poses"], d["ms_first_pose"], len(d["attempts"])), ([], None, 2))
        self.assertAlmostEqual(sum(usd), d["usd"], 6)
        m = planner.kf_message("p1", d)
        self.assertEqual((m["ok"], m["poses"]), (False, []))
        self.assertEqual(pl.stats["failed"], 1)

    def test_primary_wins_after_fallback_started(self):
        usd = []
        pl = self.mk(hedged(0.1), {PRIM: steps(0.1, 0.2), FB: [(0.05, '{"summary":"x","po'), (0.5, '"poses":[')]
                                   + steps(0.9, 0.0)[0:]})
        d = run(collect(pl, usd=usd))[-1]
        self.assertTrue(d["ok"] and d["fallback_started"] and not d["fallback_used"])
        self.assertEqual(d["model"], PRIM)
        fb = d["attempts"][1]
        self.assertTrue(fb["cancelled"] and fb["usd"] > 0 and fb["estimated"] and fb["chars"] > 0)
        self.assertEqual(len(usd), 2)

    def test_near_simultaneous_poses_never_interleave(self):
        for k in range(6):
            pl = self.mk(hedged(0.0), {PRIM: steps(0.1, 0.05 + 0.002 * k, every=0.003, n=6),
                                       FB: steps(0.9, 0.05, every=0.003, n=6)})
            evs = run(collect(pl))
            self.assertEqual(len(smiles(evs)), 1, k)
            got = [p for e in evs[:-1] for p in e["poses"]]
            self.assertEqual(len(got), 6)
            self.assertEqual(len(evs[-1]["all_poses"]), 6)

    def test_retry_without_fallback_kept(self):
        calls = []
        pl = self.mk({**NOLIB, "model": "gpt54mini", "fallback": None},
                     {PRIM: [(0.01, planner.PlanError("HTTP 429: slow down", transient=True))]}, calls)
        d = run(collect(pl))[-1]
        self.assertEqual([a["role"] for a in d["attempts"]], ["primary", "retry"])
        self.assertEqual((d["ok"], d["error"]), (False, "HTTP 429: slow down"))     # one model: its own error text

    def test_stall_is_reported_not_switched(self):
        t = pose_text(0.1, 4)
        pl = self.mk(hedged(0.05, stall_s=0.15), {PRIM: [(0.01, t[0] + t[1]), (0.3, t[2]), (0.01, t[3]), (0.01, t[4])],
                                                  FB: steps(0.9, 0.0)})
        logs = []
        pl.log = logs.append
        evs = run(collect(pl))
        d = evs[-1]
        self.assertTrue(d["ok"] and not d["fallback_started"] and d["model"] == PRIM)
        self.assertEqual(len(d["stalls"]), 1)
        self.assertTrue(d["stalls"][0]["gap_ms"] >= 280 and d["stalls"][0]["motion_s"] == 0.3, d["stalls"])
        self.assertEqual(pl.stats["stalled"], 1)
        self.assertTrue(any("stalled" in m for m in logs))
        # >= stall_motion_s of motion parsed before the gap: the page is already playing -> no stall
        pl = self.mk(hedged(0.05, stall_s=0.15), {PRIM: [(0.01, t[0] + t[1] + t[2] + t[3]), (0.3, t[4])], FB: steps(0.9, 0)})
        self.assertEqual(run(collect(pl))[-1]["stalls"], [])

    def test_session_cancel_bills_both(self):
        usd, sent, recs = [], [], []
        pl = self.mk(hedged(0.05), {PRIM: [(0.02, '{"summary":"a","poses":[')] + [(5.0, "x")],
                                    FB: [(0.02, '{"summary":"b","poses":[{"t":0.3,"smile":1}')] + [(5.0, "y")]})

        async def go():
            t = asyncio.create_task(planner.run_plan(pl, "p1", "muse", "do a thing", None, sent.append, http=object(),
                                                     on_usd=usd.append, record=recs.append))
            await asyncio.sleep(0.2)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        run(go())
        self.assertEqual(len(usd), 2)
        self.assertTrue(all(u > 0 for u in usd))
        self.assertEqual((recs[0]["ok"], recs[0]["error"]), (False, "cancelled"))

    def test_deadline_cancels_a_hung_stream(self):
        pl = self.mk({**NOLIB, "model": "gpt54mini", "fallback": None, "timeout_s": 0.3},
                     {PRIM: [(0.02, '{"poses":[{"t":0.3,"smile":1},'), (5.0, "late")]})
        t0 = time.monotonic()
        d = run(collect(pl))[-1]
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual((d["ok"], d["error"], len(d["all_poses"])), (True, "timeout", 1))

    def test_kf_messages_and_record_unchanged(self):
        sent, recs = [], []
        pl = self.mk(hedged(0.05), {PRIM: steps(0.1, 0.4), FB: steps(0.9, 0.02, n=4)})
        rec = run(planner.run_plan(pl, "p3", "muse", "do a thing", None, sent.append, http=object(),
                                   record=recs.append))
        for m in sent:
            base = ["type", "id", "poses", "done"]
            keys = base if not m["done"] else base + ["ok", "source", "engine", "summary", "ms_first_pose", "ms_done",
                                                      "usd"] + (["error"] if "error" in m else [])
            self.assertEqual(list(m), keys, m)
            for p in m["poses"]:
                self.assertEqual(list(p)[0], "t")
        self.assertTrue(all(not m["done"] for m in sent[:-1]) and sent[-1]["done"] and sent[-1]["poses"] == [])
        self.assertEqual(sum(len(m["poses"]) for m in sent), 4)
        self.assertEqual((rec["model"], rec["route"], rec["fallback_used"], rec["n_poses"]), (FB, "openrouter_zdr", True, 4))
        self.assertEqual([a["role"] for a in rec["attempts"]], ["primary", "fallback"])
        self.assertEqual(set(rec["attempts"][0]), {"model", "route", "role", "ms_start", "ms_end", "ms_first_pose",
                                                   "error", "usd", "estimated", "stalls"})
        json.dumps(rec)


# ------------------------------------------------------------------------------------------------------ cache rule
class CacheRule(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.mkdtemp(prefix="avatar-routes-cache-")

    def tearDown(self):
        shutil.rmtree(self.cache, ignore_errors=True)

    def put(self, pl, key, model, sha):
        pl.cache_put(key, {"key": key, "poses": [{"t": 0.3, "smile": 1}], "prompt_sha": sha, "model": model,
                           "summary": "s"})

    def test_acceptance(self):
        pl = planner.Planner({"library": False}, lambda m: None, cache_dir=self.cache)     # defaults: gem35lite + dsv41flash
        sha = pl.prompts["muse"]["sha256"]
        prim, fb = pl.primary()["model"], pl.fallback()["model"]
        for i, (model, served) in enumerate([("gpt-6.1-sol", True), (prim, True), (fb, True), ("z-ai/glm-5.3", False),
                                             ("gpt-5.4-mini", False)]):
            k = f"k{i}"
            self.put(pl, k, model, sha)
            self.assertEqual(pl.cache_get(k, sha, [prim, fb]) is not None, served, model)
            self.assertIsNone(pl.cache_get(k, "0" * 64, [prim, fb]))                 # another prompt: plan afresh
        pl.cfg["cache_models"] = ["glm53", "gpt-6.1-sol"]                             # presets resolve to their ids
        self.assertIsNotNone(pl.cache_get("k3", sha, [prim, fb]))
        self.assertIsNone(pl.cache_get("k0", sha, ["x"], extra=False))               # the quality pass: its own only

    def test_plan_serves_and_records_writer(self):
        calls = []
        pl = planner.Planner(hedged(0.05, cache=True), lambda m: None, cache_dir=self.cache)
        pl.stream_fn = scripted({PRIM: steps(0.1, 0.5), FB: steps(0.9, 0.01)}, calls)
        d = run(collect(pl, "hop on one foot"))[-1]
        with open(os.path.join(self.cache, d["cache_key"] + ".json")) as fh:
            ent = json.load(fh)
        self.assertEqual((ent["model"], ent["route"], ent["fallback_used"]), (FB, "openrouter_zdr", True))
        n = len(calls)
        d2 = run(collect(pl, "please hop on one foot"))
        self.assertEqual((len(d2), d2[0]["source"], d2[0]["model"], len(calls)), (1, "cache", FB, n))
        # a primary switch to a model whose plans are not welcome: plan afresh
        pl.cfg.update({"model": "dsv4pro", "fallback": None})
        pl.stream_fn = scripted({"deepseek/deepseek-v4-pro": steps(0.5, 0.01)}, calls)
        d3 = run(collect(pl, "hop on one foot"))[-1]
        self.assertEqual((d3["source"], d3["model"]), ("planner", "deepseek/deepseek-v4-pro"))

    def test_quality_fill_then_live_serves_it(self):
        calls = []
        pl = planner.Planner({"library": False}, lambda m: None, cache_dir=self.cache)
        sc = {"google/gemini-3.5-flash-lite": steps(0.2, 0.01), "gpt-6.1-sol": steps(0.7, 0.01)}
        pl.stream_fn = scripted(sc, calls)
        d = run(collect(pl, "juggle three balls"))[-1]
        self.assertEqual(d["model"], "google/gemini-3.5-flash-lite")
        q = run(pl.fill("muse", "juggle three balls"))                            # Sol replaces the lite plan
        self.assertEqual((q["source"], q["model"], q["route"], q["fallback_used"]), ("planner", "gpt-6.1-sol", "openai",
                                                                                     False))
        self.assertEqual([m for m, _ in calls], ["google/gemini-3.5-flash-lite", "gpt-6.1-sol"])
        live = run(collect(pl, "juggle 3 balls"))
        self.assertEqual((live[0]["source"], live[0]["model"]), ("cache", "gpt-6.1-sol"))
        self.assertEqual(smiles([{"done": False, "poses": [{"set": {k: v for k, v in p.items() if k != "t"}, "t": p["t"]}
                                                          for p in live[0]["poses"]]}]), {0.7})
        q2 = run(pl.fill("muse", "juggle three balls"))                           # filled already: a cache hit
        self.assertEqual((q2["source"], len(calls)), ("cache", 2))


# ------------------------------------------------------------------------------------------------------ latency
def base_planner():
    """the base commit's planner.py (5149e01), for the dispatch-overhead comparison; None if git cannot show it"""
    try:
        src = subprocess.run(["git", "-C", HERE, "show", "5149e01:app/planner.py"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:  # noqa: BLE001
        return None
    m = types.ModuleType("planner_base")
    m.__file__ = os.path.join(APP, "planner_base.py")
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


class Latency(unittest.TestCase):
    def measure(self, pl, n=300):
        http = FakeHTTP(lambda u, b: FakeResp(200, [stream_chat() if "openrouter" in u else stream_openai()]))
        out = []

        async def go():
            for i in range(n):
                t = time.perf_counter()
                async for ev in pl.plan("muse", f"do an odd thing number {i}", None, http=http):
                    if ev["done"]:
                        break
                out.append((http.calls[-1]["t"] - t) * 1e6)
        with mock.patch.dict(os.environ, DUMMY_KEYS):
            asyncio.run(go())
        return statistics.median(out[20:]), sorted(out[20:])[int(0.95 * (len(out) - 20))]

    def test_plan_start_to_request(self):
        new_or = self.measure(planner.Planner({**NOLIB, "model": "gem35lite"}, lambda m: None))
        new_oa = self.measure(planner.Planner({**NOLIB, "model": "sol", "fallback": None}, lambda m: None))
        B = base_planner()
        old = self.measure(B.Planner({**NOLIB}, lambda m: None)) if B else None
        print(f"\n  plan() start -> HTTP request issued (median / p95 us): new gem35lite+fallback {new_or[0]:.0f} / "
              f"{new_or[1]:.0f}, new sol {new_oa[0]:.0f} / {new_oa[1]:.0f}"
              + (f", base 5149e01 sol {old[0]:.0f} / {old[1]:.0f} -> added (sol vs sol) {new_oa[0] - old[0]:+.0f} us"
                 if old else ""))
        self.assertLess(new_or[0], 1000)
        if old:
            self.assertLess(new_oa[0] - old[0], 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
