"""planner mode, voice side (docs/PLANNER.md §0, §4, §5): the Realtime / GPT Live / Gemini Live
sessions of all 12 characters carry perform_action, the Muse has no `move` (its `object` stays), the timing behaviour is
described without quoted example phrases, and planner off builds the base-commit (ce9f3fd) sessions byte for byte.
No network, no paid calls.      python3 tests/test_planner_voice.py   (needs aiohttp)
"""
import ast, asyncio, importlib.util, json, os, re, subprocess, sys, tempfile, types, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "app")
sys.path.insert(0, APP)
import rtc  # noqa: E402
import gemini_live  # noqa: E402
import vendor  # noqa: E402

BASE = os.environ.get("PLANNER_BASE_COMMIT", "ce9f3fd")
A = rtc.ACTION_TOOL          # the action tool is do_action (perform_action = alias, AVATAR_ACTION_TOOL=perform_action)
PROG = ["start_behavior", "change_behavior", "stop_behavior", "list_behaviors", "look", "set_style", "learn_gesture",
        "see_camera", "attention_rule"]


def characters() -> dict:
    src = open(os.path.join(APP, "app.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CHARACTERS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("CHARACTERS not found in app.py")


CHARS = characters()


def base_module(name: str, rel: str, replace: dict | None = None):
    """the base commit's version of an app module (planner-off exactness); None if git cannot show it"""
    try:
        src = subprocess.run(["git", "-C", HERE, "show", f"{BASE}:app/{rel}"], capture_output=True, text=True, check=True).stdout
    except Exception:  # noqa: BLE001
        return None
    for a, b in (replace or {}).items():
        src = src.replace(a, b)
    m = types.ModuleType(name)
    m.__file__ = os.path.join(tempfile.gettempdir(), name + ".py")
    sys.modules[name] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__)
    return m


# the pose tool's head_pitch label was inverted (renderer + = nose down); the planner-off pin compares
# against the base commit WITH that one label corrected
RTC0 = base_module("rtc_base", "rtc.py", {'"head_pitch": "-1 .. 1 (+ = look up)"': '"head_pitch": "-1 .. 1 (+ = nose down, as in a nod; - = look up)"'})
GEM0 = base_module("gemini_base", "gemini_live.py", {"import rtc\n": "import rtc_base as rtc\n"}) if RTC0 else None

QUOTE = re.compile(r"(^|[\s(:;,])[\"“‘'][^\"”’']{2,}[\"”’']")
TIMING_KEYS = ("prepared", "about to start", "cue", "check in", "finished", "failed", "while it plays", "talk during")


def timing_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n", text) if any(k in s.lower() for k in TIMING_KEYS)]


class Sessions(unittest.TestCase):
    def test_realtime_all_characters(self):
        self.assertEqual(len(CHARS), 12)
        for key, (name, what, voice) in CHARS.items():
            for style in ("callfirst", "parallel"):
                with self.subTest(character=key, style=style):
                    s = rtc.realtime_session(name, what, voice, "server", "", style, character=key)
                    names = [t["name"] for t in s["tools"]]
                    self.assertIn(A, names)
                    self.assertNotIn("perform_action" if A == "do_action" else "do_action", names)   # one name exposed
                    for n in PROG:                                                     # every character
                        self.assertIn(n, names)
                    self.assertEqual(names.count("look"), 1)
                    self.assertNotIn("move", names)
                    self.assertEqual("object" in names, key == "muse")
                    self.assertIn(rtc.P_TIMING.strip(), s["instructions"])
                    self.assertIn(rtc.P_TALK.strip(), s["instructions"])
                    self.assertIn(rtc.P_PROGRAM.strip(), s["instructions"])
                    self.assertIn(rtc.P_WHAT_IF.strip(), s["instructions"])           # the what-if rule stays
                    if key == "muse":
                        self.assertIn("Your body's action log", s["instructions"])   # the memory rules stay
                    pa = next(t for t in s["tools"] if t["name"] == A)
                    self.assertEqual(pa["parameters"]["required"], ["description"])
                    self.assertEqual(pa["parameters"]["properties"]["talk"]["enum"], ["usual", "through", "quiet"])
                    g = next(t for t in s["tools"] if t["name"] == "gesture")
                    self.assertNotIn("jump", g["parameters"]["properties"]["name"]["enum"])
                    self.assertLess(len(json.dumps(s["tools"])), 17600)   # +9 tools, do_action.move on the muse (was < 8000); +3 quiet tools (was < 14500); tv: +tv_show/tv_clear (was < 15000); teach: +teach_board/board_erase/go_home (was < 16200); objects: +12 object verbs + relation in `object` (teach+tv tree 15830 -> merged 17232 chars; bound 17600)

    def test_live_all_characters(self):
        for key, (name, what, voice) in CHARS.items():
            with self.subTest(character=key):
                l = rtc.live_session(name, what, voice, key)
                r = l["delegation"]["responses"]
                names = [t["name"] for t in r["tools"]]
                self.assertIn(A, names)
                for n in PROG:
                    self.assertIn(n, names)
                self.assertNotIn("move", names)
                self.assertEqual("object" in names, key == "muse")
                self.assertIn(A, l["instructions"])
                self.assertIn("start_behavior", l["instructions"])
                self.assertNotIn("answer with just: done", r["instructions"])     # 'done' would read as finished

    def test_gemini_all_characters(self):
        for key, (name, what, voice) in CHARS.items():
            with self.subTest(character=key):
                sf = gemini_live.setup_for("gemini", name, what, voice, key)
                fns = [f["name"] for f in sf["setup"]["tools"][0]["functionDeclarations"]]
                self.assertIn(A, fns)
                for n in PROG:
                    self.assertIn(n, fns)
                self.assertNotIn("move", fns)
                self.assertTrue(all(len(f["description"]) <= 1000 for f in sf["setup"]["tools"][0]["functionDeclarations"]))
                instr = sf["setup"]["systemInstruction"]["parts"][0]["text"]
                self.assertIn(gemini_live.BODY_NOTE_PLANNER.strip(), instr)
                self.assertNotIn(gemini_live.BODY_NOTE.strip(), instr)
                # the thinking model makes no function calls: it keeps the planner-off session (matchers drive the body)
                st = gemini_live.setup_for("gemini-think", name, what, voice, key)
                self.assertNotIn("tools", st["setup"])
                self.assertFalse(st["planner"])

    def test_hub_relayed_live(self):
        for key, (name, what, voice) in CHARS.items():
            with self.subTest(character=key):
                tl = rtc.hub_act_tools(what, key, True)
                self.assertEqual([t["name"] for t in tl], [A] + (["object"] if key == "muse" else []) + PROG
                                 + (list(rtc.TV_TOOL_NAMES) if key == "muse" else []))     # tv: the wall TV (scene characters)
                self.assertEqual(rtc.hub_act_tools(what, key, False), rtc.act_tools() if key == "muse" else None)
                self.assertEqual(rtc.hub_act_note(what, key, False), rtc.ACT_NOTE if key == "muse" else "")

        class FakeWS:
            def __init__(self): self.sent, self.closed = [], False
            async def send_str(self, s): self.sent.append(json.loads(s))
            async def receive(self, timeout=None):
                m = types.SimpleNamespace(type=vendor.aiohttp.WSMsgType.TEXT,
                                          data=json.dumps({"type": "session.started", "session": {"id": "s", "audio": {"output": {"voice": "x"}}}}))
                return m

        class FakeHTTP:
            def __init__(self, ws): self.ws = ws
            async def ws_connect(self, *a, **k): return self.ws

        async def run(planner):
            os.environ.setdefault("OPENAI_API_KEY", "test-not-used")
            ws = FakeWS(); noop = lambda *a, **k: None
            v = vendor.LiveVendor(FakeHTTP(ws), on_audio=noop, on_transcript=noop, on_voice_usd=noop, on_backend_usd=noop,
                                  on_end=noop, log=noop, act_tools=rtc.hub_act_tools("fox", "fox", planner),
                                  act_note=rtc.hub_act_note("fox", "fox", planner), on_act=(lambda n, a: asyncio.sleep(0, {"ok": True})),
                                  planner=planner)
            await v.open("persona", "verse")
            v.reader.cancel()
            await v.prompt("about to start", True)
            await v.prompt("playing now", False)
            return ws.sent
        sent = asyncio.run(run(True))
        r = sent[0]["session"]["delegation"]["responses"]
        self.assertIn(A, [t["name"] for t in r["tools"]])
        self.assertIn(vendor.PLANNER_BODY.strip(), r["instructions"])
        self.assertNotIn("For every physical command, call animate", r["instructions"])
        self.assertEqual([e["type"] for e in sent[1:]], ["session.instructions.append", "session.thinking.append"])
        sent0 = asyncio.run(run(False))
        self.assertIn("For every physical command, call animate", sent0[0]["session"]["delegation"]["responses"]["instructions"])
        hp = rtc.hub_persona_planner("If asked to move any part of your body, generate <|SpawnThinking|> so your animation tool "
                                     "acts promptly, then describe what you did. Rest.")
        self.assertNotIn("describe what you did", hp)


@unittest.skipIf(RTC0 is None, f"git cannot show {BASE} (planner-off exactness needs the base commit)")
class PlannerOffIsTheBase(unittest.TestCase):
    def test_realtime_live_gemini_byte_for_byte(self):
        for key, (name, what, voice) in CHARS.items():
            for style in ("callfirst", "parallel"):
                with self.subTest(character=key, style=style):
                    a = rtc.realtime_session(name, what, voice, "server", "", style, character=key, planner=False)
                    b = RTC0.realtime_session(name, what, voice, "server", "", style, character=key)
                    self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
            with self.subTest(character=key, path="live"):
                self.assertEqual(json.dumps(rtc.live_session(name, what, voice, key, planner=False), sort_keys=True),
                                 json.dumps(RTC0.live_session(name, what, voice, key), sort_keys=True))
            for mode in ("gemini", "gemini-think"):
                with self.subTest(character=key, path=mode):
                    a = gemini_live.setup_for(mode, name, what, voice, key, None, False)["setup"]
                    b = GEM0.setup_for(mode, name, what, voice, key, None)["setup"]
                    self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


class NoExamplePhrases(unittest.TestCase):
    def texts(self):
        out = []
        for key, (name, what, voice) in CHARS.items():
            out.append(rtc.realtime_session(name, what, voice, "server", "", "parallel", character=key)["instructions"])
            out.append(rtc.realtime_session(name, what, voice, "server", "", "callfirst", character=key)["instructions"])
            l = rtc.live_session(name, what, voice, key)
            out += [l["instructions"], l["delegation"]["responses"]["instructions"]]
            out.append(gemini_live.setup_for("gemini", name, what, voice, key)["setup"]["systemInstruction"]["parts"][0]["text"])
            out.append(rtc.hub_act_note(what, key, True))
        out.append(rtc.hub_persona_planner("x"))
        out.append(vendor.PLANNER_BODY)
        return out

    def test_timing_sentences_have_no_quoted_phrases(self):
        n = 0
        for text in self.texts():
            for s in timing_sentences(text):
                if "action log" in s or "'suppose" in s:      # the memory rules / what-if rule (they stay as they were)
                    continue
                n += 1
                self.assertIsNone(QUOTE.search(s), s)
        self.assertGreater(n, 50)

    def test_planner_texts_have_no_quotes_at_all(self):
        prog = [x for t in rtc.program_tools() for x in [t["description"]] + [p.get("description", "") for p in t["parameters"]["properties"].values()]]
        for t in (rtc.P_TIMING, rtc.P_BODY.replace(rtc.P_WHAT_IF, ""), rtc.P_OBJECT_NOTE, *rtc.P_STYLES.values(),
                  *rtc.P_TOOL_TEXT.values(), rtc.perform_tool()["description"], rtc.P_TALK, rtc.P_PROGRAM, *prog,
                  rtc.perform_tool()["parameters"]["properties"]["description"]["description"],
                  rtc.perform_tool()["parameters"]["properties"]["talk"]["description"], vendor.PLANNER_BODY, gemini_live.BODY_NOTE_PLANNER):
            self.assertIsNone(QUOTE.search(t), t)
            self.assertNotRegex(t, r"\blike '|\be\.g\.|for example|such as '")

    def test_page_note_templates_have_no_quoted_phrases(self):
        src = ""
        for fn in ("perform.js", "voicetools.js"):                               # the voice tools' notes too
            with open(os.path.join(APP, fn), encoding="utf-8") as f:
                src += f.read()
        notes = (re.findall(r"prompt\((`[^`]*`[^,]*)", src) + re.findall(r"const NEXT_\w+ = '([^']*)'", src)
                 + re.findall(r"next: (?:[^']*)'([^']*)'", src))
        self.assertGreaterEqual(len(notes), 20)
        for n in notes:
            text = re.sub(r"\$\{[^}]*\}", "X", n.replace("`", " "))        # the words, not the template's code
            self.assertIsNone(QUOTE.search(text), n)


if __name__ == "__main__":
    unittest.main(verbosity=1)
