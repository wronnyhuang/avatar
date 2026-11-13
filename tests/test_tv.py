"""tv: unit tests for the TV language, renderer and agent.   python -m unittest tests/test_tv.py"""
import asyncio
import math
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))

import numpy as np  # noqa: E402

from tvdsl.expr import compile_expr, evaluate, ExprError, Env  # noqa: E402
from tvdsl.lang import Doc, parse_line, parse_program, ParseError, PARAM_GLIDE  # noqa: E402
from tvdsl.scene import Frame  # noqa: E402
from tvdsl.render import Renderer, rich_runs  # noqa: E402
import tv_agent  # noqa: E402


def nan(x):
    return isinstance(x, float) and x != x


class ExprSafety(unittest.TestCase):
    def test_arith(self):
        self.assertEqual(evaluate("1+2*3^2"), 19.0)
        self.assertEqual(evaluate("2^3^2"), 512.0)
        self.assertEqual(evaluate("-2^2"), -4.0)
        self.assertEqual(evaluate("(1,2)+(3,4)*2"), (7.0, 10.0))
        self.assertEqual(evaluate("t > 1 ? 5 : 6", t=2.0), 5.0)
        self.assertAlmostEqual(evaluate("lensmaker(1.5, 10, -10)"), 10.0)
        self.assertAlmostEqual(evaluate("snell(30, 1, 1.5)"), math.degrees(math.asin(0.5 / 1.5)))
        self.assertTrue(nan(evaluate("snell(60, 1.5, 1)")))           # total internal reflection

    def test_rejects_code(self):
        for bad in ['__import__("os")', "a.b", "x[0]", "lambda: 1", "foo bar", "1 +", "()", "x = 1",
                    "(" * 80 + "1" + ")" * 80, "1" * 700, "[" + ",".join(["1"] * 500) + "]"]:
            with self.assertRaises(ExprError, msg=bad):
                compile_expr(bad)

    def test_never_raises_at_eval(self):
        for src in ["1/0", "0/0", "10^1000", "exp(1e6)", "log(-1)", "sqrt(-1)", "asin(3)", "pick(nan_var, 1, 2)",
                    "at([1,2], 99)", "undefined_var * 2", "steps(0, 1, 2)", "mod(1, 0)", "unknown_func(1)",
                    "pos(nothing)", "(1,2)/(3,4)", "open('f')"]:   # a string is only ever a label word
            v = compile_expr(src).eval(Env({}))
            self.assertIsInstance(v, (float, tuple, list), src)

    def test_user_function_recursion_bounded(self):
        d = Doc()
        d.apply_text("scene \"x\"\nfunc f(x) = f(x) + 1\nlet y = f(1)\ndot a at=(0,0)", 0.0)
        fr = Frame(d, 1.0)                                      # must not hang or raise
        self.assertTrue(nan(fr.base["y"]))

    def test_const_folding(self):
        self.assertTrue(compile_expr("sin(1)*2").const)
        self.assertFalse(compile_expr("sin(t)").const)


class ParserTolerance(unittest.TestCase):
    def test_bad_lines_skipped(self):
        prog = "\n".join(["scene \"ok\"", "circle c at=(0,0) r=1", "gibberish here", "circle d at=(1,0 r=2",
                          "```", "rect r at=(0,0) w=2 h=1 color=notacolour", "line l from=(0,0) to=(1,1)"])
        stmts, errs = parse_program(prog)
        kinds = [(s.op, s.id) for s in stmts]
        self.assertIn(("obj", "c"), kinds)
        self.assertIn(("obj", "l"), kinds)
        self.assertIn(("obj", "r"), kinds)                     # a bad colour drops the property, not the object
        self.assertTrue(errs)

    def test_values(self):
        st = parse_line('arrow a from=(0, 0) to=(cos(t), sin(t)) color=cyan width=3 label="v_1" dashed')[0]
        self.assertEqual(st.kind, "arrow")
        self.assertIn("dashed", st.props)
        self.assertEqual(st.props["label"], "v_1")
        st = parse_line("plot p y=x^2 - 1 x=-2..2 color=#ff0080")[0]
        self.assertEqual(st.props["color"].const, (1.0, 0.0, 128 / 255))

    def test_for_loop(self):
        sts = parse_line("for i=0..4 dot d{i} at=(i, i^2) r=0.1")
        self.assertEqual([s.id for s in sts], ["d0", "d1", "d2", "d3", "d4"])
        self.assertEqual(sts[2].vars["i"], 2.0)
        with self.assertRaises(ParseError):
            parse_line("for i=0..100000 dot d{i}")

    def test_staged_scene_in_agent_feeder(self):
        lf = __import__("tvdsl.lang", fromlist=["LineFeeder"]).LineFeeder()
        out = lf.feed('scene "a"\ncircle c at=(0,0) r=1\nrect r at=(0,')
        self.assertEqual(len(out), 2)
        out = lf.feed("0) w=1 h=1\n")
        self.assertEqual(out[0].id, "r")


class Patches(unittest.TestCase):
    def test_param_glides(self):
        d = Doc()
        d.apply_text('scene "x"\nparam f = 2\ncircle c at=(0,0) r=f', 0.0)
        d.apply(parse_line("set f = 4")[0], 10.0)
        self.assertAlmostEqual(d.params["f"].value(10.0), 2.0)
        mid = d.params["f"].value(10.0 + PARAM_GLIDE / 2)
        self.assertTrue(2.0 < mid < 4.0)
        self.assertAlmostEqual(d.params["f"].value(10.0 + PARAM_GLIDE + 0.01), 4.0)

    def test_set_beyond_range_widens_not_clamps(self):
        d = Doc()
        d.apply_text('scene "x"\nparam L = 3 [1..4]\ncircle c at=(0,0) r=L', 0.0)
        d.apply(parse_line("set L = 6")[0], 1.0)
        self.assertAlmostEqual(d.params["L"].new, 6.0)
        self.assertEqual(d.params["L"].rng, (1.0, 6.0))
        d.apply(parse_line("param x = -1 [0..3]")[0], 2.0)
        self.assertAlmostEqual(d.params["x"].new, -1.0)

    def test_sign_flip_snaps(self):
        d = Doc()
        d.apply_text('scene "x"\nparam f = 2', 0.0)
        d.apply(parse_line("set f = -2")[0], 5.0)
        self.assertAlmostEqual(d.params["f"].value(5.01), -2.0)

    def test_replace_morphs_and_del_prefix(self):
        d = Doc()
        d.apply_text('scene "x"\ncircle c at=(0,0) r=1\nfor i=0..2 dot p{i} at=(i,0)', 0.0)
        d.apply(parse_line("circle c at=(2,0) r=1")[0], 5.0)
        self.assertIsNotNone(d.objs["c"].prev)
        fr = Frame(d, 5.0 + 0.35)
        x = fr.ref_pos("c")[0]
        self.assertTrue(0.0 < x < 2.0, x)                      # half way through the morph
        d.apply(parse_line("del p*")[0], 6.0)
        self.assertEqual([k for k in d.objs if k.startswith("p")], [])

    def test_new_scene_replaces(self):
        d = Doc()
        d.apply_text('scene "a"\ncircle c at=(0,0) r=1', 0.0)
        d.apply_text('scene "b"\nrect r at=(0,0) w=1 h=1', 1.0)
        self.assertEqual(list(d.objs), ["r"])
        self.assertEqual(d.describe(), "b")

    def test_user_func_overrides_builtin(self):
        d = Doc()
        d.apply_text('scene "x" loop=12\nfunc tri(u) = abs(mod(u,2)-1)\nlet a = tri(4.5)\nlet b = tri(t)\ndot q at=(tri(0.5),0)', 0.0)
        fr = Frame(d, 3.0)
        self.assertAlmostEqual(fr.base["a"], 0.5)
        self.assertAlmostEqual(fr.base["b"], 0.0)
        self.assertAlmostEqual(fr._geom("q")["anchor"][0], 0.5)

    def test_morph_restart_starts_the_clock_over(self):
        d = Doc()
        d.apply_text('scene "x" loop=10\ncircle c at=(0,0) r=1', 0.0)
        d.apply_text('morph "y" loop=10\ncircle c at=(1,0) r=1', 6.0)
        self.assertAlmostEqual(d.scene_t(7.0), 7.0)                  # a plain morph carries the clock on
        d.apply_text('morph "z" loop=10 restart=1\ncircle c at=(2,0) r=1', 8.0)
        self.assertAlmostEqual(d.scene_t(9.0), 1.0)                  # the careful version plays from t = 0
        self.assertIn("c", d.objs)

    def test_one_shot_traces_stay_drawn_after_the_loop_wraps(self):
        d = Doc()
        d.apply_text('scene "x" loop=4\nline a from=(0,0) to=(1,0) draw=lin(0.3,1)\n'
                     'line b from=(0,1) to=(1,1) draw=lin(0.3,1)*(t<3)\ntext c at=(0,2) "hi" alpha=ramp(0.2,0.5)', 0.0)
        st = lambda fr, k: fr._style(d.objs[k].props, fr.env_for(d.objs[k]))
        fr = Frame(d, 0.5)
        self.assertAlmostEqual(st(fr, "a")["draw"], 0.2, places=3)       # the first pass traces in
        fr = Frame(d, 4.4)                                               # t wrapped to 0.4
        self.assertAlmostEqual(st(fr, "a")["draw"], 1.0)                # ... and stays drawn
        self.assertAlmostEqual(st(fr, "c")["alpha"], 1.0)
        self.assertAlmostEqual(st(fr, "b")["draw"], 0.1, places=3)       # a stage-dependent draw loops as written

    def test_lets_dependency_order(self):
        # a morph that redefines th to use a NEW let T must not evaluate th before T (froze the pendulum, fu3)
        d = Doc()
        d.apply_text('scene "x" loop=10\nparam L = 2\nlet th = 30*cos(w*t)\nlet w = 2\nlet bob = (0,3)+polar(L,-90+th)\n'
                     'circle b at=bob r=0.3', 0.0)
        d.apply_text('morph "y"\nlet T = 2*pi*sqrt(L/9.8)\nlet th = 30*cos(2*pi*t/T)\nlet piv = (0,3)\n'
                     'let bob = piv+polar(L,-90+th)', 1.0)
        fr = Frame(d, 2.0)
        self.assertTrue(math.isfinite(fr.base["th"]), fr.base)
        x, y = fr.base["bob"]
        self.assertAlmostEqual(math.hypot(x, y - 3), 2.0, places=3)

    def test_plot_clipped_to_axes_and_lint_after_change(self):
        from tvdsl import verify
        d = Doc()
        d.apply_text('scene "g"\nparam T = 2\naxes A x=0..10 y=0..160 box=1\nplot c on=A y=2^(x/T) x=0..10', 0.0)
        before = verify.snapshot(d)
        self.assertEqual(verify.lint(d, 1.0, before), [])
        d.apply(parse_line("set T = 1")[0], 1.0)
        fr = Frame(d, 5.0)
        g = fr._geom("c")
        top = max(p[1] for m in g["marks"] for p in m.pts)
        box = fr._axes_map("A")[2]
        self.assertLessEqual(top, max(box[2], box[3]) + 0.2)          # stops at the box edge
        probs = verify.lint(d, 5.0, before)
        self.assertEqual([p[0] for p in probs], ["c"])               # this change pushed it off its axes
        self.assertIn("axes", probs[0][2])

    def test_lint_raw_template_and_off_screen(self):
        from tvdsl import verify
        d = Doc()
        d.apply_text('scene "x"\nparam m = 1\ntext a at=(0,0) "{m>0.5 ? \'big\' : \'small\'}"\n'
                     'text b at=(0,1) "{{m>0.5 ? "big" : "small"}}"\ncircle far at=(0,9) r=0.5\ncircle ok at=(0,0) r=1', 0.0)
        probs = {p[0]: p[2] for p in verify.lint(d, 1.0)}
        self.assertEqual(set(probs), {"b", "far"})
        self.assertIn("raw", probs["b"])
        self.assertIn("off the screen", probs["far"])

    def test_flow_ring_of_long_names_fits(self):
        for names in ('["magma","igneous rock","sediment","sedimentary rock","metamorphic rock"]',
                      '["acetyl CoA","citrate","isocitrate","alpha ketoglutarate","succinyl CoA","succinate","fumarate","malate"]'):
            d = Doc()
            d.apply_text(f'scene "c"\nflow F at=(0,-0.3) nodes={names} ring=1 active=0', 0.0)
            R = Renderer(512, 288)
            x0, x1, y0, y1 = R.fit_view(d, 1.0)
            self.assertGreaterEqual(x0, -8.6)
            self.assertLessEqual(x1, 8.6)
            self.assertGreaterEqual(min(y0, y1), -4.9)
            self.assertLessEqual(max(y0, y1), 4.9)

    def test_program_text_roundtrip(self):
        src = 'scene "a" loop=8\nparam f = 2 0.5..4\nlet g = f*2\ncircle c at=(0,0) r=g\nfor i=0..2 dot p{i} at=(i,0)'
        d = Doc()
        d.apply_text(src, 0.0)
        d2 = Doc()
        d2.apply_text(d.program_text(), 0.0)
        self.assertEqual(set(d.objs), set(d2.objs))
        self.assertEqual(d2.params["f"].new, 2.0)


class Optics(unittest.TestCase):
    def frame(self, src):
        d = Doc()
        d.apply_text(src, 0.0)
        return d, Frame(d, 10.0)

    def axis_cross(self, pts):
        for a, b in zip(pts, pts[1:]):
            if a[1] * b[1] < 0 and a[0] > -0.01:
                return a[0] + (b[0] - a[0]) * (a[1] / (a[1] - b[1]))
        return None

    def test_parallel_rays_meet_at_focus(self):
        for f in (1.5, 2.5, 4.0):
            d, fr = self.frame(f'scene "x"\nlens L at=(0,0) f={f} h=3')
            for y in (0.5, 1.2, -2.0):
                pts, _, _ = fr.trace((-6.0, y), (1.0, 0.0))
                self.assertAlmostEqual(self.axis_cross(pts), f, places=6)

    def test_point_images_through_lens(self):
        d, fr = self.frame('scene "x"\nlens L at=(0,0) f=2 h=3')
        do, h = 5.0, 1.0
        di = 1 / (1 / 2 - 1 / do)
        for ang in (-10.0, -5.0, 0.0, 3.0):
            pts, _, _ = fr.trace((-do, h), (math.cos(math.radians(ang)), math.sin(math.radians(ang))))
            a, b = pts[1], pts[2]
            y = a[1] + (b[1] - a[1]) * (di - a[0]) / (b[0] - a[0])
            self.assertAlmostEqual(y, -h * di / do, places=6)

    def test_lensmaker_lens(self):
        d, fr = self.frame('scene "x"\nlens L at=(0,0) r1=3 r2=-3 n=1.5 h=2')
        g = fr._geom("L")
        self.assertAlmostEqual(g["f"], 3.0)

    def test_snell_and_tir(self):
        d, fr = self.frame('scene "x"\nglass g pts=[(-8,0),(8,0),(8,-4),(-8,-4)] n=1.5')
        a = 30.0
        pts, _, _ = fr.trace((-math.sin(math.radians(a)) * 3, math.cos(math.radians(a)) * 3),
                             (math.sin(math.radians(a)), -math.cos(math.radians(a))))
        p, q = pts[1], pts[2]
        b = math.degrees(math.atan2(q[0] - p[0], p[1] - q[1]))
        self.assertAlmostEqual(math.sin(math.radians(a)), 1.5 * math.sin(math.radians(b)), places=6)
        # from inside, past the critical angle: reflects back down
        d, fr = self.frame('scene "x"\nglass g pts=[(-8,0),(8,0),(8,-4),(-8,-4)] n=1.5')
        pts, _, _ = fr.trace((-2.0, -2.0), (math.sin(math.radians(60)), math.cos(math.radians(60))))
        self.assertLess(pts[2][1], pts[1][1])

    def test_mirror_focus(self):
        d, fr = self.frame('scene "x"\nmirror M at=(3,0) f=2 h=3')
        pts, _, _ = fr.trace((-6.0, 1.0), (1.0, 0.0))
        a, b = pts[1], pts[2]
        x = a[0] + (b[0] - a[0]) * (0 - a[1]) / (b[1] - a[1])
        self.assertAlmostEqual(x, 1.0, places=6)


class Render(unittest.TestCase):
    PROG = "\n".join([
        'scene "test" loop=8', "param f = 2", "axes x=-7..7 y=-4..4 ticks=1 grid=1", "lens L at=(0,0) f=f h=2 axis=1 marks=2",
        "arrow o from=(-5,0) to=(-5,1.2) color=green label=\"object\"", "rays3 r from=(-5,1.2) lens=L draw=lin(0,2)",
        "image i of=o lens=L", "beam b from=(-7,-2) dir=10 n=5 h=1 white=1", "plot p y=sin(x+t) x=-6..6 color=cyan",
        "for k=0..20 dot d{k} at=(-6+k*0.6, 3+0.3*sin(t+k)) r=0.08 color=hsv(k*17,0.7,1)",
        'text e at=(0,3.7) "1/f = (n−1)(1/R_1 − 1/R_2)  f = {f:.2f}"', "atom O at=(5,-2.5) el=O shells=[2,6]",
        "wave w from=(-7,-3.5) to=(7,-3.5) amp=0.3 wl=1.5 speed=1"])

    def test_deterministic(self):
        d = Doc()
        d.apply_text(self.PROG, 0.0)
        a = Renderer(512, 288).render(d, 3.3)
        b = Renderer(512, 288).render(d, 3.3)
        self.assertEqual(a.shape, (288, 512, 3))
        self.assertTrue(np.array_equal(a, b))
        self.assertGreater(int(a.max()), 100)                   # something visible

    def test_frame_budget(self):
        d = Doc()
        d.apply_text(self.PROG, 0.0)
        R = Renderer(512, 288)
        R.render(d, 0.1)
        ts = []
        for k in range(30):
            t0 = time.perf_counter()
            R.render(d, 1.0 + k / 12)
            ts.append(time.perf_counter() - t0)
        ts.sort()
        # p50 inside a 12 fps frame even for this busy test scene on a loaded hub (typical scenes: ~15 ms)
        self.assertLess(ts[len(ts) // 2], 1 / 12 * 0.9, ts[len(ts) // 2])

    def test_rich_text(self):
        self.assertEqual(rich_runs("H_2O"), [("H", 0), ("2", -1), ("O", 0)])
        self.assertEqual(rich_runs("e^[-x]"), [("e", 0), ("−x", 1)])

    def test_garbage_does_not_crash(self):
        d = Doc()
        d.apply_text('scene "x"\ncircle c at=(1e308,1e308) r=1e308\nline l from=(0,0) to=(inf,0)\npath p pts=[]\n'
                     'text t at=(0,0) "{1/0}"\nray r from=(0,0) dir=nan\nplot p2 y=1/x x=-1..1', 0.0)
        Renderer(512, 288).render(d, 1.0)


class FakeDoor:
    """streams scripted text per call; delays per call index"""

    def __init__(self, scripts, first_delay=None, chunk=12):
        self.scripts = list(scripts)
        self.first_delay = first_delay or [0.0] * len(scripts)
        self.calls = 0
        self.chunk = chunk

    def prewarm(self, *a, **k):
        pass

    async def stream(self, model="haiku", system="", user="", images=None, max_s=20.0, tag=""):
        k = self.calls
        self.calls += 1
        text = self.scripts[min(k, len(self.scripts) - 1)]
        await asyncio.sleep(self.first_delay[min(k, len(self.first_delay) - 1)])
        for i in range(0, len(text), self.chunk):
            yield {"delta": text[i:i + self.chunk]}
            await asyncio.sleep(0.005)
        yield {"done": True, "ok": True, "text": text, "ms_first": 1, "ms_done": 2, "error": None}


PROG = 'scene "a lens"\nparam f = 2\nlens L at=(0,0) f=f h=2\nray r from=(-5,1) dir=0\n'


class Agent(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def agent(self, door, **cfg):
        c = {"render": False, "hedge_s": [], "tap_debounce_s": 0.05, "tap_min_gap_s": 0.0, "upgrade": None}
        c.update(cfg)
        states = []
        return tv_agent.TVAgent(cfg=c, door=door, on_state=states.append), states

    def test_show_and_patch(self):
        async def go():
            door = FakeDoor([PROG, "set f = 1\nnote \"shorter focal length\"\n"])
            tv, states = self.agent(door)
            r = tv.show("a lens")
            self.assertTrue(r["ok"])
            await tv.cur["task"]
            self.assertEqual(tv.doc.title, "a lens")
            self.assertEqual(tv.scene_id, 1)
            tv.hear("you", "can you make the focal length shorter please")
            await asyncio.sleep(0.2)
            await tv.cur["task"]
            self.assertEqual(tv.doc.params["f"].new, 1.0)
            self.assertEqual(tv.describe(), "shorter focal length")
            st = [s["status"] for s in states]
            self.assertIn("shown", st)
            self.assertIn("updated", st)
            self.assertEqual(states[-1]["n_patches"], 1)
            self.assertEqual(states[-1]["phase"], "showing")
            await tv.close()
        self.run_async(go())

    def test_bad_stream_keeps_last_picture(self):
        async def go():
            door = FakeDoor([PROG, 'scene "broken"\nthis is not tvl\nneither is this\n'])
            tv, states = self.agent(door)
            tv.show("a lens")
            await tv.cur["task"]
            tv.show("something else")
            await tv.cur["task"]
            self.assertEqual(tv.doc.title, "a lens")             # the new scene never committed
            self.assertEqual(states[-1]["status"], "failed")
            await tv.close()
        self.run_async(go())

    def test_none_and_backchannel(self):
        async def go():
            door = FakeDoor([PROG, "none\n"])
            tv, states = self.agent(door)
            tv.show("a lens")
            await tv.cur["task"]
            tv.hear("you", "okay")                               # backchannel: no call
            await asyncio.sleep(0.1)
            self.assertEqual(door.calls, 1)
            tv.hear("you", "that is really neat, thanks a lot")
            await asyncio.sleep(0.2)
            await tv.cur["task"]
            self.assertEqual(door.calls, 2)
            self.assertEqual(tv.history[-1]["status"], "unchanged")
            await tv.close()
        self.run_async(go())

    def test_new_request_supersedes(self):
        async def go():
            door = FakeDoor([PROG, 'scene "second"\ncircle c at=(0,0) r=1\n'], first_delay=[0.5, 0.0])
            tv, states = self.agent(door)
            tv.show("first")
            await asyncio.sleep(0.05)
            tv.show("second")
            await tv.cur["task"]
            await asyncio.sleep(0.6)
            self.assertEqual(tv.doc.title, "second")
            self.assertEqual(tv.history[0]["status"], "cancelled")
            await tv.close()
        self.run_async(go())

    def test_hedge_takes_first_speaker(self):
        async def go():
            door = FakeDoor(['scene "slow"\ncircle c at=(0,0) r=1\n', 'scene "fast"\ncircle c at=(0,0) r=1\n'],
                            first_delay=[1.0, 0.0])
            tv, states = self.agent(door, hedge_s=[0.1])
            t0 = time.monotonic()
            tv.show("x")
            await tv.cur["task"]
            self.assertEqual(tv.doc.title, "fast")
            self.assertLess(time.monotonic() - t0, 0.8)
            await tv.close()
        self.run_async(go())

    def test_clear(self):
        async def go():
            tv, states = self.agent(FakeDoor([PROG]))
            tv.show("x")
            await tv.cur["task"]
            tv.clear()
            self.assertTrue(tv.doc.empty)
            self.assertEqual(states[-1]["phase"], "idle")
            await tv.close()
        self.run_async(go())


class Surface:
    def __init__(self):
        self.frames = []
        self.cleared = 0

    def put_frame(self, rgb, t=None):
        self.frames.append(np.array(rgb))

    def clear(self, t=None):
        self.cleared += 1


class Process(unittest.TestCase):
    def test_worker_process_frames(self):
        async def go():
            S = Surface()
            tv = tv_agent.TVAgent(surface=S, cfg={"process": True, "fps": 12, "hedge_s": [], "upgrade": None}, door=FakeDoor([PROG]))
            tv.show("x")
            await tv.cur["task"]
            t0 = time.monotonic()
            while not S.frames and time.monotonic() - t0 < 8:
                await asyncio.sleep(0.05)
            self.assertTrue(S.frames, "no frame from the render process")
            self.assertEqual(S.frames[-1].shape, (288, 512, 3))
            # the loop stays responsive while it renders
            lag = []
            for _ in range(20):
                a = time.monotonic()
                await asyncio.sleep(0.01)
                lag.append(time.monotonic() - a - 0.01)
            self.assertLess(max(lag), 0.03)
            tv.clear()
            await asyncio.sleep(0.4)
            self.assertGreaterEqual(S.cleared, 1)
            await tv.close()
        asyncio.new_event_loop().run_until_complete(go())


if __name__ == "__main__":
    unittest.main()


class Round4(unittest.TestCase):
    """tv successor: root(), 3D, auto sweep, tick labels, inner reflection"""

    def test_root(self):
        from tvdsl.expr import evaluate
        self.assertAlmostEqual(evaluate("root(x^2-2, 0, 3)"), 2 ** 0.5, 6)
        self.assertAlmostEqual(evaluate("root(10-x-(2+x), 0, 10)"), 4.0, 6)
        self.assertTrue(evaluate("root(x^2+1, 0, 3)") != evaluate("root(x^2+1, 0, 3)"))      # no root -> NaN

    def test_vsepr_tetrahedral(self):
        import math as m
        from tvdsl.expr import evaluate
        a = [evaluate("vsepr(4,%d)" % i) for i in range(4)]
        for k in (1, 2, 3):
            self.assertAlmostEqual(m.degrees(m.acos(sum(x * y for x, y in zip(a[0], a[k])))), 109.47, 1)
        self.assertEqual(evaluate("p3(0,0,1)", yaw=0, pitch=0), (0.0, 0.0))

    def test_sweep(self):
        d = Doc()
        d.apply_text('scene "p"\nparam a = 1 [-2..3] sweep\nplot p y=a*x^2 x=-3..3', 0.0)
        self.assertEqual(d.sweep_param(), "a")
        d0 = Doc()
        d0.apply_text('scene "p"\nparam a = 1 [-2..3]\nplot p y=a*x^2 x=-3..3', 0.0)
        self.assertIsNone(d0.sweep_param())                     # opt-in only
        self.assertAlmostEqual(d.sweep_value("a", 2.0), 1.0)
        self.assertAlmostEqual(d.sweep_value("a", 3.5 + 6.0), -2.0, 5)
        d.apply_text("set a = 2", 20.0)
        self.assertIsNone(d.sweep_param())                       # he set it: the TV stops exploring by itself
        d2 = Doc()
        d2.apply_text('scene "w"\nparam k = 1 [0..3] sweep\nplot p y=sin(k*x-t) x=-3..3', 0.0)
        self.assertIsNone(d2.sweep_param())                      # it already moves

    def test_ticks_stay_ordered(self):
        d = Doc()
        d.apply_text('scene "c"\naxes A x=0..10 y=0..100 box=1 labels="q,p"\ntext big at=(0,0) "a long label right here"', 0.0)
        img = Renderer(512, 288).render(d, 2.0)
        self.assertEqual(img.shape, (288, 512, 3))

    def test_raindrop_inner(self):
        from tvdsl.scene import Frame
        d = Doc()
        d.apply_text('scene "r"\nglass g at=(0,0) r=2 n=1.33 inner=1\nray r from=(-7,1.5) dir=0', 0.0)
        fr = Frame(d, 1.0)
        pts, _, _ = fr.trace((-7, 1.5), (1, 0))
        self.assertGreaterEqual(len(pts), 4)                     # in, reflect at the back, out
        self.assertLess(pts[-1][0], pts[2][0])                   # leaves heading back toward the light



class OffLoopEncode(unittest.TestCase):
    """tv: frames are diffed + encoded on the pump thread; the hub loop only stamps and sends"""

    def test_surface_gets_frames_and_mirror_matches(self):
        import surfaces as SF
        sent = []
        S = SF.Surfaces(send=sent.append)
        tv_s = S.get("tv")
        on_loop = []

        async def go():
            ag = tv_agent.TVAgent(surface=lambda: tv_s, cfg={"model": "haiku", "process": False})
            orig = tv_s.commit_frame

            def timed(enc, t=None):
                t0 = time.perf_counter()
                r = orig(enc, t)
                on_loop.append(time.perf_counter() - t0)
                return r
            tv_s.commit_frame = timed
            ag._send_lines(['scene "x" loop=4', 'circle c at=(2*osc(2),0) r=1.5 color=cyan width=3', 'text l at=(0,3) "hello"'])
            await asyncio.sleep(1.5)
            await ag.close()
        asyncio.run(go())
        self.assertGreater(len(sent), 5)
        self.assertGreater(int(tv_s.canvas[:, :, :3].max()), 100)
        mirror = np.zeros_like(tv_s.canvas)
        for m in sent:
            d = SF.unpack_msg(m)
            if d["w"]:
                mirror[d["y"]:d["y"] + d["h"], d["x"]:d["x"] + d["w"]] = d["pixels"]
        # JPEG is lossy: the receiver's mirror is close to the hub canvas
        self.assertLess(float(np.abs(mirror[:, :, :3].astype(int) - tv_s.canvas[:, :, :3].astype(int)).mean()), 3.0)
        on_loop.sort()
        self.assertLess(on_loop[len(on_loop) // 2], 0.003)          # the loop's share is cheap


class Strings(unittest.TestCase):
    def test_conditional_words_in_text(self):
        from tvdsl.scene import interp
        self.assertEqual(interp("{a<0 ? 'opens down' : 'opens up'}", Env({"a": 2.0})), "opens up")
        d = Doc()
        d.apply_text("scene \"s\"\nparam th = 50\ntext t at=(0,0) \"{th>41.8 ? 'TIR' : 'refracts'}\"", 0.0)
        self.assertEqual([m.text for m in Frame(d, 1.0).marks() if m.kind == "text"], ["TIR"])


class UpdateSafety(unittest.TestCase):
    def test_hold_stops_sweep_and_glides_back(self):
        d = Doc()
        d.apply_text('scene "p"\nparam a = 1 [-2..3] sweep\nplot p y=a*x^2 x=-3..3', 0.0)
        from tvdsl.scene import Frame
        v_sw = Frame(d, 9.5).base["a"]
        self.assertLess(v_sw, 0.0)                                # being swept toward -2
        d.apply_text("hold", 9.5)
        self.assertAlmostEqual(Frame(d, 9.5).base["a"], v_sw, 5)  # no jump
        self.assertAlmostEqual(Frame(d, 9.5 + PARAM_GLIDE + 0.1).base["a"], 1.0, 5)   # glides back, sweep over

    def test_update_fragment_with_scene_line_is_a_patch(self):
        prog0 = ['scene "sine"', "let a = 2", "axes x=-6..6 y=-2..2", "plot s y=sin(a*x) x=-6..6 color=cyan",
                 "dot p at=(1,sin(a)) r=0.1"]
        patch = ['scene "doubled"', "plot s y=sin(2*a*x) x=-6..6 color=cyan", 'text t at=(0,3) "sin 2x"']
        states = []

        async def go():
            ag = tv_agent.TVAgent(cfg={"render": False, "hedge_s": [], "review": None, "upgrade": None}, door=FakeDoor(["\n".join(prog0), "\n".join(patch)]),
                                  on_state=states.append)
            ag.show("sine")
            while ag.busy():
                await asyncio.sleep(0.01)
            ag._start("update", "x", "tap", "double it")
            while ag.busy():
                await asyncio.sleep(0.01)
            return ag
        ag = asyncio.run(go())
        self.assertIn("p", ag.doc.objs)                           # the old picture's objects survived
        self.assertEqual(ag.scene_id, 1)
        self.assertEqual(ag.doc.note, "doubled")


class LoopLen(unittest.TestCase):
    def test_loop_covers_the_longest_sequence(self):
        d = Doc()
        d.apply_text('scene "s" loop=10\nlet a = cycle(2.5, 10, 25, 40, 50, 60, 75)\nline l from=(0,0) to=polar(3,a)', 0.0)
        self.assertAlmostEqual(d.loop_len(), 16.5)
        d2 = Doc()
        d2.apply_text('scene "s" loop=10\nline l from=(0,0) to=(1,1)', 0.0)
        self.assertEqual(d2.loop_len(), 10.0)


class Upgrade(unittest.TestCase):
    SKETCH = 'scene "sketch"\ncircle c at=(0,0) r=1\n'
    CAREFUL = 'scene "careful"\ncircle c at=(0,0) r=1\nline a from=(0,0) to=(2,0)\nline b from=(0,0) to=(0,2)\ndot d at=(1,1)\n'

    def run_case(self, delays, tap=None):
        states = []

        async def go():
            ag = tv_agent.TVAgent(cfg={"render": False, "hedge_s": [], "upgrade": "sonnet", "model": "haiku",
                                       "tap_debounce_s": 0.05, "tap_min_gap_s": 0.0, "verify": False},
                                  door=FakeDoor([self.SKETCH, self.CAREFUL, "set x = 1\nnote \"tapped\""], first_delay=delays),
                                  on_state=states.append)
            ag.show("something")
            if tap:
                await asyncio.sleep(0.15)
                ag.hear("you", tap)
            for _ in range(300):
                await asyncio.sleep(0.02)
                if ag.settled():
                    await asyncio.sleep(0.05)
                    if ag.settled():
                        break
            return ag
        return asyncio.run(go()), states

    def test_careful_version_replaces_sketch(self):
        ag, states = self.run_case([0.0, 0.3])
        self.assertEqual(ag.doc.title, "careful")
        self.assertIn("upgraded", [h["status"] for h in ag.history])
        self.assertEqual(states[-1]["source"], "upgrade")

    def test_tap_does_not_wait_for_the_careful_version(self):
        # tv: his change lands at once on the sketch (it used to wait ~4 s for the careful version); the
        # careful version in flight is dropped and the careful pass is redone on the changed picture
        ag, states = self.run_case([0.0, 0.8, 0.0], tap="make the circle a lot bigger please")
        self.assertEqual(ag.doc.title, "sketch")
        self.assertEqual(ag.doc.note, "tapped")
        self.assertIn("cancelled", [h["status"] for h in ag.history if h["kind"] == "upgrade"])
        self.assertIn("check", [h["kind"] for h in ag.history])     # the careful pass looked at the changed sketch

    def test_careful_version_morphs_in(self):
        ag, states = self.run_case([0.0, 0.3])
        lines = [l for _, l in ag.timeline]
        self.assertTrue(any(l.startswith("morph") for l in lines))    # no cut: the sketch's circle c morphs
        self.assertFalse(any(l.startswith("scene") for l in lines[2:]))


class FollowUp(unittest.TestCase):
    """tv: a follow-up must change the DRAWING (verified by rendering before/after), else it is escalated"""
    LENS = 'scene "lens"\nparam R = 3 [1..6]\nlens L at=(0,0) r1=R r2=-R n=1.5 h=2.2 axis=1\nbeam b from=(-7,0) dir=0 n=5 h=1.6\ntext eq at=(0,3.8) "f = 3"\n'

    def go(self, scripts, words, **cfg):
        door = FakeDoor(scripts)
        c = {"render": False, "hedge_s": [], "upgrade": None, "careful": None, "tap_debounce_s": 0.02,
             "tap_min_gap_s": 0.0}
        c.update(cfg)
        states = []

        async def run():
            ag = tv_agent.TVAgent(cfg=c, door=door, on_state=states.append)
            ag.show("a lens")
            for _ in range(200):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            ag.hear("you", words)
            await asyncio.sleep(0.1)
            for _ in range(400):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            return ag
        return asyncio.run(run()), door, states

    def test_param_patch_passes_the_check(self):
        ag, door, states = self.go([self.LENS, 'note "more curved"\nset R = 1.5'], "make the lens more curved please")
        up = [h for h in ag.history if h["kind"] == "update"][-1]
        self.assertGreater(up["check"]["geom"], 0.0015)
        self.assertNotIn("fix", up)
        self.assertEqual(up["change"], "more curved")
        self.assertEqual(states[-1].get("change"), "more curved")     # the voice model is told what changed
        self.assertEqual(door.calls, 2)

    def test_hedge_patch_is_corrected_by_the_strong_patch(self):
        # call 1 (strong, primary) starts slowly; the hedge (call 2) wins with a weak patch; the primary's patch then
        # lands as a correction on top of the picture as it was BEFORE his words
        door = FakeDoor([self.LENS, 'note "lens surfaces more curved"\nset R = 1.4\ntext eq at=(0,3.8) "smaller R"',
                         'note "curved"\nset R = 2.5\nrect junk at=(5,0) w=1 h=1'], first_delay=[0.0, 0.6, 0.0])

        async def run():
            ag = tv_agent.TVAgent(cfg={"render": False, "hedge_s": [], "upgrade": None, "careful": None,
                                       "tap_debounce_s": 0.02, "tap_min_gap_s": 0.0, "patch_hedge_s": [0.1]}, door=door)
            ag.show("a lens")
            for _ in range(100):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            ag.hear("you", "make the lens more curved please")
            await asyncio.sleep(0.4)
            mid = (ag.doc.params["R"].new, "junk" in ag.doc.objs)
            for _ in range(200):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            return ag, mid
        ag, mid = asyncio.run(run())
        self.assertEqual(mid, (2.5, True))                          # the quick patch showed first
        self.assertAlmostEqual(ag.doc.params["R"].new, 1.4)         # then the strong one
        self.assertNotIn("junk", ag.doc.objs)
        up = [h for h in ag.history if h["kind"] == "update"][-1]
        self.assertEqual(up.get("winner_model"), "haiku")
        cor = [h for h in ag.history if h["kind"] == "correct"][-1]
        self.assertEqual(cor["status"], "corrected")
        self.assertEqual(ag.changes[-1][1], "lens surfaces more curved")
        self.assertEqual(door.calls, 3)

    def test_caption_only_patch_is_escalated(self):
        ag, door, states = self.go([self.LENS, 'note "more curved"\ntext eq at=(0,3.8) "smaller R, shorter f"',
                                    'note "lens surfaces more curved"\nset R = 1.4'], "make the lens more curved please")
        up = [h for h in ag.history if h["kind"] == "update"][-1]
        self.assertEqual(up["fix"]["status"], "applied")
        self.assertGreater(up["check"]["geom"], 0.0015)
        self.assertAlmostEqual(ag.doc.params["R"].new, 1.4)
        self.assertEqual(door.calls, 3)

    def test_rewrite_is_a_morph(self):
        prog = 'scene "lens, rebuilt"\nparam R = 1.5 [1..6]\nlens L at=(0,0) r1=R r2=-R n=1.5 h=2.4 axis=1\nbeam b from=(-7,0) dir=0 n=3 h=1.6\nrect x at=(5,0) w=1 h=1\ntext t2 at=(0,-3.8) "R = {R}"\n'
        ag, door, states = self.go([self.LENS, prog], "rebuild it with a much more curved lens and a block")
        lines = [l for _, l in ag.timeline]
        self.assertTrue(any(l.startswith("morph") for l in lines))
        self.assertTrue(any(l.startswith("del") and "eq" in l for l in lines))   # what the new program lacks fades out
        self.assertIn("x", ag.doc.objs)
        self.assertNotIn("eq", ag.doc.objs)

    def test_tool_update_reuses_the_tap(self):
        async def run():
            door = FakeDoor([self.LENS, 'note "more curved"\nset R = 1.5'], first_delay=[0.0, 0.3])
            ag = tv_agent.TVAgent(cfg={"render": False, "hedge_s": [], "upgrade": None, "careful": None,
                                       "tap_debounce_s": 0.02, "tap_min_gap_s": 0.0}, door=door)
            ag.show("a lens")
            for _ in range(100):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            ag.hear("you", "make the lens more curved please")
            await asyncio.sleep(0.15)                       # the tap's patch is in flight
            r = ag.show("make the lens more curved", mode="update")
            for _ in range(200):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            return ag, door, r
        ag, door, r = asyncio.run(run())
        self.assertEqual(r["status"], "updating")
        self.assertEqual(door.calls, 2)                     # no second patch for the same words

    def test_edit_mode_checks_the_sketch_by_looking(self):
        class LookDoor(FakeDoor):
            async def stream(self, **kw):
                self.images = getattr(self, "images", []) + [kw.get("images")]
                async for ev in FakeDoor.stream(self, **kw):
                    yield ev
        door = LookDoor([self.LENS, 'note "fixed"\ntext eq at=(0,3.8) "f = {R:.1f}"'])

        async def run():
            ag = tv_agent.TVAgent(cfg={"render": False, "hedge_s": [], "careful": "edit", "check_model": "sonnet"}, door=door)
            ag.show("a lens")
            for _ in range(300):
                await asyncio.sleep(0.02)
                if ag.settled() and any(h["kind"] == "check" for h in ag.history):
                    break
            return ag
        ag = asyncio.run(run())
        chk = [h for h in ag.history if h["kind"] == "check"]
        self.assertEqual(chk[0]["status"], "checked")
        self.assertTrue(door.images[1] and os.path.exists(door.images[1][0]))   # it was shown a rendered still
        self.assertEqual(ag.careful_state, "done")


class LoopHold(unittest.TestCase):
    """tv: after a change the loop does not wrap (the picture would blank and re-trace) for LOOP_HOLD s"""

    def test_hold_delays_the_wrap_then_restarts(self):
        d = Doc()
        for ln in ['scene "x" loop=4', 'circle c at=(0,0) r=1 draw=lin(0,1)']:
            for st in parse_line(ln):
                d.apply(st, 100.0)
        self.assertAlmostEqual(d.scene_t(105.0), 1.0)              # wraps normally
        for st in parse_line("hold"):
            d.apply(st, 103.5)
        self.assertAlmostEqual(d.scene_t(104.5), 4.5)              # no wrap while he looks at the change
        self.assertAlmostEqual(d.scene_t(103.5 + d.LOOP_HOLD + 0.5), 0.5)   # then the loop starts over
        for st in parse_line('morph "y" loop=6'):
            d.apply(st, 120.0)
        self.assertEqual(d.title, "y")
        self.assertIn("c", d.objs)                                 # a morph keeps the objects


class Lint(unittest.TestCase):
    def test_undefined_name_is_found_and_fixed(self):
        from tvdsl import verify
        d = Doc()
        for ln in ['scene "x"', 'circle c at=(0,0) r=1', 'text w at=(0,3) "f = {fz:.2f}"', 'line bad from=(q,0) to=(q,2)']:
            for st in parse_line(ln):
                d.apply(st, 10.0)
        probs = verify.lint(d, 10.0)
        self.assertEqual([p[0] for p in probs], ["w"])
        door = FakeDoor(['scene "x"\nparam R = 2\ncircle c at=(0,0) r=R\ntext w at=(0,3) "f = {fz:.2f}"\n',
                         'set R = 3', 'text w at=(0,3) "f = {R/2:.2f}"'])

        async def run():
            ag = tv_agent.TVAgent(cfg={"render": False, "hedge_s": [], "upgrade": None, "careful": None,
                                       "tap_debounce_s": 0.02, "tap_min_gap_s": 0.0, "verify": False}, door=door)
            ag.show("x")
            for _ in range(100):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            ag.hear("you", "make it a bit bigger please")
            await asyncio.sleep(0.1)
            for _ in range(200):
                await asyncio.sleep(0.02)
                if ag.settled():
                    break
            return ag
        ag = asyncio.run(run())
        up = [h for h in ag.history if h["kind"] == "update"][-1]
        self.assertEqual(up.get("lint"), ["w"])
        self.assertIn("R/2", ag.doc.objs["w"].src)


class AnimatedParam(unittest.TestCase):
    def test_animated_param_returns_to_him_on_a_change(self):
        d = Doc()
        for ln in ['scene "lens" loop=10', 'param n = 1.5 [1.2..2]', 'set n = cycle(4,1.5,1.9,1.3)', 'circle c at=(0,0) r=n']:
            for st in parse_line(ln):
                d.apply(st, 0.0)
        self.assertNotIn("n", d.params)                   # animated: a let now
        for st in parse_line("hold"):
            d.apply(st, 5.0)
        self.assertIn("n", d.params)                      # his again, gliding back to 1.5
        self.assertAlmostEqual(d.params["n"].value(5.0 + PARAM_GLIDE + 0.01), 1.5)
        self.assertNotIn("n", dict(d.lets))


# ---------------------------------------------------------------------------------------------------- tv-prim
def _doc(prog, now=0.0):
    d = Doc()
    errs = d.apply_text(prog, now)
    assert not errs, errs
    return d


def _overlap(a, b, slack=1):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw - slack and bx < ax + aw - slack and ay < by + bh - slack and by < ay + ah - slack


def _cross(p1, p2, q1, q2):
    def o(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return o(p1, p2, q1) * o(p1, p2, q2) < 0 and o(q1, q2, p1) * o(q1, q2, p2) < 0


def _scene_eval(fr, src):
    return compile_expr(src).eval(Env(dict(fr.base), fr))


def _centre(pts):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


class TextLayout(unittest.TestCase):
    def render(self, prog, t=3.0, w=512, h=288):
        R = Renderer(w, h, autofit=False)
        img = R.render(_doc(prog), t)
        return R, img

    def test_legibility_floor_scales_with_width(self):
        prog = 'scene "x" view=-40,40,-22.5,22.5\ntext a at=(0,0) "small words" size=0.45'
        for w, h in ((512, 288), (1024, 576)):
            R, _ = self.render(prog, w=w, h=h)
            (_, x, y, tw, th, _s), = R.last_texts
            self.assertGreaterEqual(th, 12 * w / 512, (w, th))        # 0.45 world at 6.4 px/unit would be ~3 px

    def test_long_text_wraps_and_stays_on_screen(self):
        R, _ = self.render('scene "x"\ntext a at=(0,3.8) "' + "a fairly long caption " * 6 + '" size=0.6')
        (txt, x, y, w, h, _s), = R.last_texts
        self.assertIn("\n", txt)                                       # wrapped at spaces, not shrunk to dust
        self.assertTrue(0 <= x and x + w <= 512 and 0 <= y and y + h <= 288, (x, y, w, h))
        self.assertGreaterEqual(h / (txt.count("\n") + 1), 12)

    def test_free_texts_never_overlap(self):
        R, _ = self.render('scene "x"\ntext a at=(0,0) "first label"\ntext b at=(0.2,0.1) "second label"\n'
                           'text c at=(-0.2,-0.1) "third label"')
        boxes = [(x, y, w, h) for _t, x, y, w, h, _s in R.last_texts]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(_overlap(boxes[i], boxes[j], 3), (boxes[i], boxes[j]))

    def test_free_text_slides_off_a_line_when_space_exists(self):
        R, _ = self.render('scene "x"\nline l from=(-8,0) to=(8,0) color=white width=3\ntext a at=(0,0) "on the line"')
        (_t, x, y, w, h, _s), = R.last_texts
        self.assertFalse(y + 2 < 144 < y + h - 2, (y, h))             # the line (y = 144 px) no longer runs through it

    def test_text_inside_a_filled_box_stays(self):
        R, _ = self.render('scene "x"\nrect r at=(0,0) w=5 h=2 fill=teal\ntext a at=(0,0) "inside"')
        (_t, x, y, w, h, _s), = R.last_texts
        self.assertLess(abs((x + w / 2) - 256), 3)
        self.assertLess(abs((y + h / 2) - 144), 3)

    def test_axis_numbers_thin_out_instead_of_colliding(self):
        R, _ = self.render('scene "x" view=-20,20,-11.25,11.25\naxes x=-18..18 y=-10..10 ticks=1')
        xs = sorted((x, w) for t, x, y, w, h, s in R.last_texts if s == "tick" and abs(y + h / 2 - 144) < 20 and w > 0)
        self.assertGreater(len(xs), 4)
        for (x0, w0), (x1, _w1) in zip(xs, xs[1:]):
            self.assertGreater(x1, x0 + w0 - 2)

    def test_compact_big_tick_numbers(self):
        from tvdsl.scene import _fmtnum
        self.assertEqual(_fmtnum(250000.0), "250k")
        self.assertEqual(_fmtnum(1250000.0), "1.25M")
        self.assertEqual(_fmtnum(-3.0), "−3")
        self.assertEqual(_fmtnum(2030.0), "2030")

    def test_plot_label_at_the_right_end(self):
        R, _ = self.render('scene "x"\nplot p y=0.3*x x=-6..4 color=lime label="line"')
        (_t, x, y, w, h, _s), = R.last_texts
        ex, ey = 256 + 4 * 32, 144 - 1.2 * 32                          # the curve's right end in pixels
        self.assertLess(math.hypot(x - ex, (y + h / 2) - ey), 60)


class Callouts(unittest.TestCase):
    PROG = '''scene "cell"
ellipse mem at=(0,0) rx=4 ry=2.8 color=lime
circle nuc at=(-0.6,0) r=1.1 fill=purple
callout c1 at=(-0.6,0.9) "nucleus"
callout c2 at=(-0.4,0.2) "nucleolus"
callout c3 at=(2.2,-1.2) "mitochondrion"
callout c4 at=(2.4,1) "endoplasmic reticulum"
callout c5 at=(3.95,0.3) "cell membrane"
callout c6 at=(0.5,-2.2) "cytoplasm"
text ttl at=(0,3.9) "Animal cell"'''

    def test_parse_and_marks(self):
        fr = Frame(_doc(self.PROG), 1.0)
        cm = [m for m in fr.marks() if m.kind == "text" and m.soft == "callout"]
        self.assertEqual(len(cm), 6)
        self.assertEqual(fr.ref_pos("c3"), (2.2, -1.2))
        st = parse_line('callout k at=(1,2) "x" side=left')[0]
        self.assertEqual((st.kind, st.props["side"]), ("callout", "left"))
        self.assertEqual(parse_line('annotate k at=(1,2) "x"')[0].kind, "callout")

    def test_labels_beside_the_picture_no_overlap_no_crossing(self):
        R = Renderer(512, 288, autofit=False)
        R.render(_doc(self.PROG), 2.0)
        calls = [(t, x, y, w, h) for t, x, y, w, h, s in R.last_texts if s == "callout"]
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(R.last_leaders), 6)
        boxes = [(x, y, w, h) for _t, x, y, w, h, _s in R.last_texts]
        for i in range(len(boxes)):
            self.assertTrue(0 <= boxes[i][0] and boxes[i][0] + boxes[i][2] <= 512)
            for j in range(i + 1, len(boxes)):
                self.assertFalse(_overlap(boxes[i], boxes[j], 3), (R.last_texts[i], R.last_texts[j]))
        L = R.last_leaders
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                self.assertFalse(_cross(L[i][0], L[i][1], L[j][0], L[j][1]), (L[i], L[j]))
        # the membrane ellipse spans x 128..384 px: labels sit in the margins, not on the cell
        for _t, x, y, w, h in calls:
            self.assertTrue(x + w <= 130 or x >= 382, (_t, x, w))

    def test_explicit_side(self):
        R = Renderer(512, 288, autofit=False)
        R.render(_doc('scene "x"\ncircle c at=(0,0) r=1\ncallout a at=(0,1) "top" side=t\n'
                      'callout b at=(0,-1) "bottom" side=b'), 2.0)
        pos = {t: (y, h) for t, x, y, w, h, s in R.last_texts}
        self.assertLess(pos["top"][0] + pos["top"][1], 144 - 32)
        self.assertGreater(pos["bottom"][0], 144 + 32)


class Flows(unittest.TestCase):
    def test_ring_positions_and_no_overlap(self):
        d = _doc('scene "x"\nflow W at=(0,0) nodes=["evaporation","condensation","precipitation","collection"] ring=1 '
                 'active=floor(t/2)%4\ndot s at=node(W,0)+(0,1)')
        fr = Frame(d, 3.0)
        n = [fr.ref_node("W", i) for i in range(4)]
        self.assertAlmostEqual(n[0][0], 0.0, 6)
        self.assertGreater(n[0][1], 1.0)                        # step 0 on top, then clockwise
        self.assertGreater(n[1][0], 1.0)
        self.assertLess(n[2][1], -1.0)
        self.assertLess(n[3][0], -1.0)
        self.assertEqual(fr.ref_pos("s"), (n[0][0], n[0][1] + 1))
        self.assertEqual(_scene_eval(fr, "node(W, 5)"), n[1])  # wraps around the ring
        boxes = [m for m in fr.marks() if m.kind == "path" and m.closed and m.fill is not None and m.stroke is None
                 and max(p[0] for p in m.pts) - min(p[0] for p in m.pts) > 1.0]      # (not the dots)
        self.assertEqual(len(boxes), 4)
        bb = []
        for m in boxes:
            xs, ys = [p[0] for p in m.pts], [p[1] for p in m.pts]
            bb.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(_overlap(bb[i], bb[j], 0), (bb[i], bb[j]))
        # active step 1 (t=3): white outline, and a dot on the arrow out of it
        strokes = [m for m in fr.marks() if m.kind == "path" and m.closed and m.stroke == (1.0, 1.0, 1.0)]
        self.assertEqual(len(strokes), 1)
        self.assertLess(math.dist(_centre(strokes[0].pts), n[1]), 1e-6)
        arrows = [m for m in fr.marks() if m.kind == "path" and m.head == 1]
        self.assertEqual(len(arrows), 4)                        # a ring closes the loop

    def test_row_and_column(self):
        fr = Frame(_doc('scene "x"\nflow R at=(0,1) nodes=["a","b","c"] row=1\n'
                        'flow C at=(4,0) nodes=["x","y"] col=1 color=lime'), 1.0)
        r = [fr.ref_node("R", i) for i in range(3)]
        self.assertTrue(r[0][0] < r[1][0] < r[2][0] and r[0][1] == r[1][1] == 1.0)
        c = [fr.ref_node("C", i) for i in range(2)]
        self.assertTrue(c[0][1] > c[1][1] and c[0][0] == c[1][0] == 4.0)
        self.assertEqual(len([m for m in fr.marks() if m.kind == "path" and m.head == 1]), 3)   # 2 + 1: not closed

    def test_long_row_fits_the_screen(self):
        fr = Frame(_doc('scene "x"\nflow R at=(0,0) nodes=["source code","lexer","parser","code generator",'
                        '"machine code"] row=1'), 1.0)
        xs = [p[0] for m in fr.marks() if m.kind == "path" and m.closed for p in m.pts]
        self.assertGreater(min(xs), -8.0)
        self.assertLess(max(xs), 8.0)
        self.assertTrue(any("\n" in m.text for m in fr.marks() if m.kind == "text"))   # long names wrapped

    def test_renders(self):
        img = Renderer(512, 288).render(_doc('scene "x"\nflow F nodes=["one","two","three"] active=t%3'), 1.5)
        self.assertGreater(int(img.max()), 100)


class Waves(unittest.TestCase):
    def test_ripple_crests(self):
        d = _doc('scene "x" loop=30\nripple R at=(1,0) wl=0.8 speed=1.5 r=6')
        fr = Frame(d, 2.0)
        (m,) = [m for m in fr.marks() if m.segs]
        radii = sorted(math.dist(seg[0], (1.0, 0.0)) for seg, _a in m.segs)
        self.assertLessEqual(radii[-1], 1.5 * 2.0 + 1e-6)              # nothing beyond the front
        for a, b in zip(radii, radii[1:]):
            self.assertAlmostEqual(b - a, 0.8, 4)                     # one wavelength apart
        self.assertAlmostEqual(min((radii[-1] - 3.0) % 0.8, 0.8 - (radii[-1] - 3.0) % 0.8), 0.0, 4)  # moving at `speed`
        strength = [a for _s, a in sorted(m.segs, key=lambda sa: math.dist(sa[0][0], (1.0, 0.0)))]
        self.assertGreater(strength[0], strength[-1])                  # fade with distance
        self.assertFalse([m for m in Frame(_doc('scene "x"\nripple R at=(0,0) start=5'), 1.0).marks() if m.segs])
        img = Renderer(512, 288).render(_doc('scene "x"\nripple A at=(-1,0)\nripple B at=(1,0)'), 4.0)
        self.assertGreater(int(img.max()), 80)

    def test_standing_wave(self):
        d = _doc('scene "x" loop=100\nwave w from=(-4,0) to=(4,0) amp=0.5 wl=2 speed=1 mode=standing nodes=1')
        # cos(2 pi speed t / wl) = 0 at t = 0.5: the string is flat; at t = 1 it is a full (inverted) 2*amp loop
        flat = Frame(d, 0.5).marks()
        path = max((m for m in flat if m.kind == "path"), key=lambda m: len(m.pts))
        self.assertLess(max(abs(p[1]) for p in path.pts), 1e-6)
        full = max((m for m in Frame(d, 1.0).marks() if m.kind == "path"), key=lambda m: len(m.pts))
        self.assertAlmostEqual(max(abs(p[1]) for p in full.pts), 1.0, 2)
        nodes = [m for m in flat if m.kind == "path" and m.closed]
        self.assertEqual(len(nodes), 9)                                # every wl/2 over 8 units, both ends included
        self.assertIsNotNone(parse_line("wave w standing")[0].props.get("standing"))

    def test_slit_intensities(self):
        d, L, wl, a = 0.25, 1000.0, 0.00055, 0.06
        self.assertAlmostEqual(evaluate("slit2(0, 0.25, 1000, 0.00055)"), 1.0)
        y1 = evaluate("fringe(1, 0.25, 1000, 0.00055)")
        self.assertAlmostEqual(y1, L * math.tan(math.asin(wl / d)), 6)
        self.assertAlmostEqual(evaluate(f"slit2({y1}, 0.25, 1000, 0.00055)"), 1.0, 6)    # bright fringe
        ymin = L * math.tan(math.asin(wl / (2 * d)))
        self.assertLess(evaluate(f"slit2({ymin}, 0.25, 1000, 0.00055)"), 1e-9)          # dark fringe
        yenv = L * math.tan(math.asin(wl / a))
        self.assertLess(evaluate(f"slit1({yenv}, 0.06, 1000, 0.00055)"), 1e-9)          # single-slit zero
        self.assertLess(evaluate(f"slit2({y1}, 0.25, 1000, 0.00055, 0.06)"), 1.0)       # under the envelope
        self.assertAlmostEqual(evaluate("grating(0, 0.25, 1000, 0.00055, 6)"), 1.0)
        self.assertTrue(nan(evaluate("fringe(5, 1, 10, 0.5)")))                         # no 5th order


class PrimPerf(unittest.TestCase):
    def test_busy_frame(self):
        prog = '''scene "busy" loop=12
axes A x=0..10 y=0..100 box=1 labels="time,speed" grid=1
plot p1 on=A y=8*x+5 x=0..10 color=lime label="fast"
plot p2 on=A y=x^2 x=0..10 color=orange label="square"
for i=0..9 dot d{i} on=A at=(i, 10*i) r=0.1 color=yellow
for i=0..5 callout k{i} at=(-3+1.2*i,-1+0.4*i) "part {i}"
flow F at=(0,0) nodes=["one","two","three","four"] ring=1 active=t/2%4
ripple R1 at=(-4,1) wl=0.8 r=10
ripple R2 at=(-4,-1) wl=0.8 r=10
wave w from=(-7,-2) to=(7,-2) mode=standing nodes=1'''
        R = Renderer(512, 288)
        d = _doc(prog)
        for k in range(3):
            R.render(d, 5 + k * 0.1)
        ts = []
        for k in range(8):
            t0 = time.perf_counter()
            R.render(d, 6 + k * 0.083)
            ts.append(time.perf_counter() - t0)
        self.assertLess(sorted(ts)[4], 0.15)            # generous for a loaded box; ~30 ms measured
