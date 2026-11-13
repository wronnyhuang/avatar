"""TEACH — the server side of the teaching performance (seam S4).

One TeachHub per session (Session.teach). The page's director (app/teach.js) asks for a board plan; the board brain
(app/board_brain.py, seam S3) streams items back as {type:'board_item'}; the page performs them and reports the chalk
tip's marks ({type:'board_ink', t, segs}) which are drawn into the session's `board` surface (seam S2) stamped on the
playback timeline, so a mark appears when the rendered chalk gets there. Erases the same way (discs under the felt).

page -> hub   teach_plan {id, request, context, board}        teach_erase {id, what, context, board}
              teach_cancel {id, why}                         board_ink {t, id, segs:[{item, stroke, s0, s1, pts, width,
              color} | {erase:true, pts:[[u,v]], ru, rv}]}    board_item_done {id, item, ok}   board_erased {id, item}
              teach_utterance {text}  (the user's words while it teaches: topic change -> teach_abort)
hub -> page   board_item {id, item}   board_plan_done {id, n, error?}   board_plan_error {id, error}
              board_state {summary, items}   teach_abort {why}

Nothing here is subject-specific: the brain decides what to write, code decides the strokes and the timing.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

CHALK = (240, 240, 232, 235)
WIDTH_V = float(os.environ.get("TEACH_CHALK_WIDTH_V", "0.011"))   # chalk line width, v units (fraction of the board height)
TOPIC_SYS = ("You decide whether a listener's words, said while a teacher is writing on a board about a topic, mean the "
             "lesson should STOP (the listener says to stop or leave it, or changes to an unrelated subject) or CONTINUE "
             "(a question or remark about the lesson, agreement, small talk that does not change the subject). Answer "
             "with one word: STOP or CONTINUE.")
ERASE_RESIDUE = float(os.environ.get("TEACH_ERASE_RESIDUE", "0.03"))   # chalk smear the felt leaves (0 = clean)
DEICTIC = re.compile(r"\s*(?:all of )?(?:that|this|it|those|these|them|that one|this one|what you (?:just )?wrote)\s*[.!]?\s*", re.I)
STOP_HINT = re.compile(r"\b(stop|never ?mind|forget (it|about it|that)|come back|go back|enough|cancel)\b", re.I)


def _rgba(c):
    if isinstance(c, (list, tuple)) and len(c) in (3, 4):
        try:
            v = [max(0, min(255, int(x))) for x in c]
            return tuple(v + [235] if len(v) == 3 else v)
        except (TypeError, ValueError):
            return CHALK
    named = {"white": CHALK, "yellow": (246, 226, 120, 235), "red": (236, 120, 110, 235), "blue": (140, 190, 246, 235),
             "green": (150, 226, 150, 235), "pink": (246, 160, 200, 235), "orange": (246, 180, 110, 235)}
    return named.get(str(c or "").lower(), CHALK)


class _Recorder:
    """Stand-in surface (before the scene's surfaces exist, and in tests): records the ops."""

    def __init__(self):
        self.ops = []

    def draw_polyline(self, pts, width, rgba=CHALK, t=None, ts=None):
        self.ops.append(("line", pts, width, rgba, t))

    def erase_disc(self, c, r, t=None, residue=0.08):
        self.ops.append(("erase", c, r, t))

    def erase_rect(self, a, b, t=None, residue=0.08):
        self.ops.append(("erase_rect", a, b, t))

    def clear(self, t=None):
        self.ops.append(("clear", t))


def _stub_strokes(text):
    """A tiny single-stroke stub font (digits, + - = x and a box for anything else) — only used when the board brain
    is not importable (tests, an unmerged branch)."""
    G = {"0": [[(.3, 0), (.05, .2), (.05, .8), (.3, 1), (.55, .8), (.55, .2), (.3, 0)]], "1": [[(.15, .2), (.35, 0), (.35, 1)]],
         "2": [[(.05, .2), (.3, 0), (.55, .2), (.55, .4), (.05, 1), (.6, 1)]], "3": [[(.05, .1), (.3, 0), (.55, .2), (.3, .5), (.55, .75), (.3, 1), (.05, .9)]],
         "4": [[(.45, 1), (.45, 0), (.05, .7), (.6, .7)]], "+": [[(.3, .25), (.3, .85)], [(0, .55), (.6, .55)]],
         "=": [[(.05, .4), (.55, .4)], [(.05, .7), (.55, .7)]], "-": [[(.05, .55), (.55, .55)]], "x": [[(.05, .35), (.55, 1)], [(.55, .35), (.05, 1)]]}
    return G


class TeachHub:
    def __init__(self, session, surfaces=None, brain=None, door=None, log=None):
        self.s = session
        self._surfaces = surfaces
        self.brain_mod = brain
        self.door = door
        self.log = log or (lambda *a: None)
        self.tasks: dict[str, asyncio.Task] = {}
        self.bsess = None                 # board_brain.BoardSession (the board model across requests)
        self.items: dict[str, dict] = {}  # what is on the board (performed), id -> {id, kind, text, bbox}
        self.pending: dict[str, dict] = {}   # planned, not yet performed
        self.n = {"plans": 0, "items": 0, "ink_msgs": 0, "ink_pts": 0, "erase_pts": 0, "cancels": 0, "aborts": 0}
        self.first_item_ms: list[int] = []
        self.active_req = None
        self.last_utt_check = 0.0
        self.prewarmed = False
        # the part of the board the chalk reaches (page: places.board.station.reach_uv_y, the board's aspect): the brain
        # lays out on that band as its whole board (aspect = band width / band height), items are mapped onto the board
        self.band = (0.0, 1.0)
        self.aspect = 2.0
        self.raw: dict[str, dict] = {}    # item id -> the brain's own item (band coordinates) for mark_done

    def prewarm(self):
        """A session that may teach (its scene has a board): keep the board brain's door processes warm (seam S1/S3)."""
        if self.prewarmed:
            return
        self.prewarmed = True
        try:                                   # the change-of-subject judge (the user's words mid-lesson) on a warm process too
            from claude_door import door as _d  # noqa: PLC0415
            _d.prewarm("haiku", TOPIC_SYS)
        except Exception as e:  # noqa: BLE001
            self.log(f"teach topic prewarm failed: {e}")
        B = self._brain()
        if B is not None and hasattr(B, "prewarm"):
            try:
                B.prewarm()
            except Exception as e:  # noqa: BLE001
                self.log(f"teach prewarm failed: {e}")

    # -- surfaces: the session's (scene seam S2) or a recorder
    def surface(self, name="board"):
        S = self._surfaces if self._surfaces is not None else getattr(self.s, "surfaces", None)
        if S is None:
            S = self._surfaces = {"board": _Recorder(), "tv": _Recorder()}
        try:
            return S.get(name)
        except Exception:  # noqa: BLE001
            return None

    def send(self, m):
        try:
            self.s.send(m)
        except Exception as e:  # noqa: BLE001
            self.log(f"teach send failed: {e}")

    def _brain(self):
        if self.brain_mod is None:
            try:
                import board_brain  # noqa: PLC0415  (seam S3; absent before the boardbrain merge)
                self.brain_mod = board_brain
            except Exception:  # noqa: BLE001
                self.brain_mod = False
        return self.brain_mod or None

    def _session(self):
        B = self._brain()
        if B is not None and self.bsess is None and hasattr(B, "BoardSession"):
            try:
                self.bsess = B.BoardSession(aspect=self.aspect)
            except TypeError:
                self.bsess = B.BoardSession()
        return self.bsess

    # -- messages from the page
    def on_asr(self, text: str, final: bool, asr: str = "full") -> None:
        """the server's streaming recognizer (vosk, partials within ~0.3 s of the words) for the lesson's stop
        words: a short utterance with a stop hint goes to the page as {type:'teach_heard'}; the page's teach.userText
        decides (only while a lesson / station visit is on, never on a negated stop). The realtime voice's own
        transcript arrives 2-4 s later and is checked there as before."""
        t = (text or "").strip()
        if not t or len(t.split()) > 9 or not STOP_HINT.search(t):
            return
        key = (asr, t.lower())
        if key == getattr(self, "_asr_last", None):
            return
        self._asr_last = key
        try:
            self.s.send({"type": "teach_heard", "text": t[:200], "final": bool(final), "asr": asr})
        except Exception:  # noqa: BLE001
            pass

    def on_msg(self, m: dict) -> bool:
        t = m.get("type")
        if t == "teach_hello":
            self.prewarm()
        elif t == "teach_plan" or t == "teach_erase":
            self._area(m.get("area"))
            self._start(m, erase=(t == "teach_erase"))
        elif t == "teach_cancel":
            rid = str(m.get("id") or "")
            task = self.tasks.pop(rid, None)
            if task and not task.done():
                task.cancel()
                self.n["cancels"] += 1
            self._brain_abort()
            if self.active_req == rid:
                self.active_req = None
        elif t == "board_ink":
            self._ink(m)
        elif t == "board_item_done":
            it = self.pending.pop(str(m.get("item") or ""), None)
            if it is not None and m.get("ok", True) is not False:
                self.items[it["id"]] = it
            self._brain_done(str(m.get("item") or ""))
            self._state()
        elif t == "board_erased":
            if not m.get("partial"):          # a lazy page turn's columns: the brain hears of it once, at the end
                self._brain_done(str(m.get("item") or ""))
            self._erased(m)
        elif t == "teach_utterance":
            self._utterance(m)
        else:
            return False
        return True

    def _start(self, m: dict, erase: bool):
        rid = str(m.get("id") or "")[:40]
        for k, task in list(self.tasks.items()):       # one plan at a time: a new request replaces the old one
            if not task.done():
                task.cancel()
            self.tasks.pop(k, None)
        self.active_req = rid
        self.n["plans"] += 1
        task = asyncio.create_task(self._plan(rid, m, erase))
        self.tasks[rid] = task
        task.add_done_callback(lambda t, rid=rid: self.tasks.pop(rid, None) if self.tasks.get(rid) is t else None)

    async def _plan(self, rid: str, m: dict, erase: bool):
        t0 = time.monotonic()
        ctx = [x for x in (m.get("context") or []) if isinstance(x, dict)][-12:]
        request = str(m.get("what") if erase else m.get("request") or "")[:400]
        n = 0

        async def on_item(it):
            nonlocal n
            if self.active_req != rid:
                return
            it = _jsonable(it)
            self.raw[str(it.get("id"))] = it
            it = self._to_board(it)
            if n == 0:
                self.first_item_ms.append(int((time.monotonic() - t0) * 1000))
            n += 1
            self.n["items"] += 1
            if it.get("kind") not in ("erase", "say"):
                self.pending[str(it.get("id"))] = {"id": str(it.get("id")), "kind": it.get("kind"), "text": str(it.get("text") or "")[:80],
                                                   "bbox": it.get("bbox"), "plan": self.n["plans"]}
            self.send({"type": "board_item", "id": rid, "item": it})

        try:
            last = self._last_lesson() if erase and DEICTIC.fullmatch(request) else None
            if last:
                # "erase that / it / this" right after a lesson = that lesson's items: no brain call (2-4 s)
                ids = [i["id"] for i in last]
                regs = [[round(max(0.0, b[0] - 0.012), 4), round(max(0.0, b[1] - 0.012), 4), round(min(1.0, b[2] + 0.012), 4), round(min(1.0, b[3] + 0.012), 4)]
                        for b in (i["bbox"] for i in last)]
                await on_item({"id": "erase_last", "kind": "erase", "regions": regs, "target": ids, "say_before": None, "strokes": []})
                if self.bsess is not None:
                    try:
                        if hasattr(self.bsess, "board"):
                            self.bsess.board.erase_ids(list(ids))
                        if hasattr(self.bsess, "mark_done"):
                            self.bsess.mark_done({"id": "erase_last", "kind": "erase", "target": ids})
                    except Exception as e:  # noqa: BLE001
                        self.log(f"teach erase-last brain update failed: {e}")
                self.raw.pop("erase_last", None)
            elif erase and re.fullmatch(r"\s*(all|everything|the (whole )?board|board|it all|all of it)\s*", request, re.I):
                await on_item({"id": "erase_all", "kind": "erase", "regions": [[0, 0, 1, 1]], "say_before": None, "strokes": []})
                if self.bsess is not None:
                    try:
                        self.bsess.clear() if hasattr(self.bsess, "clear") else self.bsess.board.erase_all()
                    except Exception:  # noqa: BLE001
                        pass
            else:
                S = self._session()
                if S is not None:
                    req = ("Erase from the board: " + request) if erase else request
                    await S.plan(ctx, req, on_item=on_item)
                else:
                    await self._stub(request, on_item, erase)
            self.send({"type": "board_plan_done", "id": rid, "n": n, "ms": int((time.monotonic() - t0) * 1000)})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.log(f"teach plan failed: {type(e).__name__}: {e}")
            self.send({"type": "board_plan_error", "id": rid, "error": f"{type(e).__name__}"})

    async def _stub(self, request, on_item, erase):
        if erase:
            await on_item({"id": "e1", "kind": "erase", "regions": [[0, 0, 0.5, 1]], "strokes": []})
            return
        G = _stub_strokes(request)
        expr = "".join(ch for ch in request if ch in G) or "2+2=4"
        parts = re.split(r"(?==)", expr)
        v = 0.12
        for i, part in enumerate(parts):
            strokes, u = [], 0.06
            for ch in part:
                for st in G.get(ch, []):
                    strokes.append([[round(u + x * 0.09, 4), round(v + y * 0.18, 4)] for x, y in st])
                u += 0.08
            await asyncio.sleep(0.4)
            await on_item({"id": f"s{i}", "kind": "math", "say_before": "First this." if i == 0 else "And so.", "strokes": strokes,
                           "bbox": [0.06, v, u, v + 0.18], "text": part, "est_s": 2.0})
            v += 0.26

    def _last_lesson(self):
        """the items of the most recent lesson still on the board (what 'that' points at right after a lesson)"""
        its = [i for i in self.items.values() if isinstance(i.get("bbox"), (list, tuple)) and len(i["bbox"]) == 4 and i.get("plan") is not None]
        if not its:
            return None
        p = max(i["plan"] for i in its)
        return [i for i in its if i["plan"] == p]

    # -- the reach band: brain coordinates (0..1 of the band) <-> board coordinates (0..1 of the writable area)
    def _area(self, a):
        if not isinstance(a, dict):
            return
        try:
            v0, v1 = float(a.get("v0", 0)), float(a.get("v1", 1))
            asp = float(a.get("aspect") or 2.0)
        except (TypeError, ValueError):
            return
        if not (0 <= v0 < v1 <= 1 and v1 - v0 > 0.2 and 0.5 < asp < 8):
            return
        if self._brain_reach():
            # the board brain lays out inside the reach band itself (board.layout.Board reach=, BOARD_REACH_V): full-board
            # coordinates both ways, nothing to map; it learns the band from the scene through the env switch
            want = f"{v0:.4g},{v1:.4g}"
            if os.environ.get("BOARD_REACH_V") != want and not self.items:
                os.environ["BOARD_REACH_V"] = want
                self.bsess = None
            self.band, self.aspect = (0.0, 1.0), round(asp, 4)
            return
        band, aspect = (round(v0, 4), round(v1, 4)), round(asp / (v1 - v0), 4)
        if band != self.band and self.bsess is not None and self.items:
            return                                    # a board already laid out keeps its band
        if band != self.band or aspect != self.aspect:
            self.band, self.aspect = band, aspect
            if not self.items:
                self.bsess = None                     # rebuilt on the next plan with the band's aspect

    def _brain_reach(self) -> bool:
        """does the board brain keep its marks inside a reach band itself?"""
        if getattr(self, "_reach_ok", None) is None:
            ok = False
            try:
                import inspect  # noqa: PLC0415
                from board.layout import Board  # noqa: PLC0415
                ok = "reach" in inspect.signature(Board.__init__).parameters
            except Exception:  # noqa: BLE001
                ok = False
            self._reach_ok = ok
        return self._reach_ok

    def _to_board(self, it):
        v0, v1 = self.band
        if (v0, v1) == (0.0, 1.0):
            return it
        k = v1 - v0
        mv = lambda v: round(v0 + float(v) * k, 5)  # noqa: E731
        out = dict(it)
        if isinstance(it.get("strokes"), list):
            out["strokes"] = [[[float(p[0]), mv(p[1])] for p in st if isinstance(p, (list, tuple)) and len(p) >= 2]
                              for st in it["strokes"] if isinstance(st, (list, tuple))]
        b = it.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out["bbox"] = [b[0], mv(b[1]), b[2], mv(b[3])]
        if isinstance(it.get("regions"), list):        # [0,0,1,1] = 'the whole board' (the page wipes where chalk is)
            out["regions"] = [list(r) if list(r) == [0, 0, 1, 1] else [r[0], mv(r[1]), r[2], mv(r[3])]
                              for r in it["regions"] if isinstance(r, (list, tuple)) and len(r) == 4]
        for key in ("size", "erase_r"):
            if isinstance(it.get(key), (int, float)):
                out[key] = round(float(it[key]) * k, 5)
        if isinstance(it.get("erase_wh"), (list, tuple)) and len(it["erase_wh"]) == 2:
            out["erase_wh"] = [it["erase_wh"][0], round(float(it["erase_wh"][1]) * k, 5)]
        return out

    def _brain_done(self, iid):
        it = self.raw.pop(iid, None)
        if it is not None and self.bsess is not None and hasattr(self.bsess, "mark_done"):
            try:
                self.bsess.mark_done(it)
            except Exception as e:  # noqa: BLE001
                self.log(f"teach mark_done failed: {e}")

    def _brain_abort(self):
        """a plan cut short: the brain forgets what was planned but never written (its board = what is really there)"""
        self.pending.clear()
        if self.bsess is not None and hasattr(self.bsess, "abort"):
            try:
                if hasattr(self.bsess, "tracking"):
                    self.bsess.tracking = True       # nothing written yet is still a fact: an empty plan leaves nothing
                self.bsess.abort()
            except Exception as e:  # noqa: BLE001
                self.log(f"teach brain abort failed: {e}")
        self.raw.clear()

    # -- the marks
    def _ink(self, m: dict):
        B = self.surface("board")
        if B is None:
            return
        t = m.get("t")
        t = float(t) if isinstance(t, (int, float)) else None
        self.n["ink_msgs"] += 1
        for sg in (m.get("segs") or [])[:200]:
            if not isinstance(sg, dict):
                continue
            pts = [(float(p[0]), float(p[1])) for p in (sg.get("pts") or [])[:400] if isinstance(p, (list, tuple)) and len(p) >= 2]
            if not pts:
                continue
            try:
                if sg.get("erase"):
                    r = float(sg.get("rv") or 0.05)
                    ru = sg.get("ru")
                    if sg.get("sweep") and len(pts) >= 2 and isinstance(ru, (int, float)) and ru > 0:
                        # a swept stamp = the felt's track over one frame (a fast stroke moves several
                        # felt widths a frame): rects along the segment, spaced well inside the pad
                        hu, hv = max(0.005, min(0.2, float(ru) * 0.8)), max(0.005, min(0.2, r))
                        (a0, b0), (a1, b1) = pts[0], pts[1]
                        n = max(1, min(60, int(max(abs(a1 - a0) / hu, abs(b1 - b0) / hv) * 1.25) + 1))
                        pts = [(a0 + (a1 - a0) * i / n, b0 + (b1 - b0) * i / n) for i in range(n + 1)]
                    for c in pts:
                        if isinstance(ru, (int, float)) and ru > 0:
                            # the felt is a pad, not a disc: a rectangle ru*0.8 x rv half-size (a disc of
                            # radius rv was only 0.027 uv wide: 12% of the chalk survived a full-board wipe)
                            hu, hv = max(0.005, min(0.2, float(ru) * 0.8)), max(0.005, min(0.2, r))
                            B.erase_rect((c[0] - hu, c[1] - hv), (c[0] + hu, c[1] + hv), t=t, residue=ERASE_RESIDUE)
                        else:
                            B.erase_disc(c, max(0.01, min(0.2, r)), t=t, residue=ERASE_RESIDUE)
                    self.n["erase_pts"] += len(pts)
                else:
                    w = sg.get("width")
                    B.draw_polyline(pts, float(w) if isinstance(w, (int, float)) and 0 < w < 0.2 else WIDTH_V, rgba=_rgba(sg.get("color")), t=t)
                    self.n["ink_pts"] += len(pts)
            except Exception as e:  # noqa: BLE001
                self.log(f"teach ink failed: {e}")

    def _erased(self, m: dict):
        regions = [r for r in (m.get("regions") or []) if isinstance(r, (list, tuple)) and len(r) == 4]
        gone = []
        for iid, it in list(self.items.items()):
            b = it.get("bbox")
            if not b or len(b) != 4:
                continue
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in regions):
                gone.append(iid)
        tg = str(m.get("item") or "")
        # the wipe is done: what the felt could not quite reach inside its regions goes too (live, ~1-10% of
        # the chalk stayed as ghosts and the layout then wrote over them: r2 fractions), so the board matches its state
        B = self.surface("board")
        if B is not None and m.get("ok") is not False:
            t = m.get("t")
            t = float(t) if isinstance(t, (int, float)) else None
            for r in regions:
                try:
                    B.erase_rect((float(r[0]) - 0.005, float(r[1]) - 0.01), (float(r[2]) + 0.005, float(r[3]) + 0.01), t=t, residue=0.0)
                except Exception as e:  # noqa: BLE001
                    self.log(f"teach clean-up failed: {e}")
        for iid in gone:
            self.items.pop(iid, None)
        self.log(f"teach erased {tg}: {gone}")
        self._state()

    def _state(self):
        items = list(self.items.values())
        self.send({"type": "board_state", "summary": summarize(items), "items": items})

    # -- the user's words while it teaches: a clear change of subject ends the lesson (harness-enforced, not left to the voice
    # model). A cheap classifier via claude_door (the `claude` CLI); no door = only the page's stop words apply.
    def _utterance(self, m: dict):
        text = str(m.get("text") or "")[:300].strip()
        if not text or not self.active_req and not m.get("teaching"):
            return
        door = self.door
        if door is None:
            try:
                from claude_door import door as _d  # noqa: PLC0415  (seam S1)
                door = self.door = _d
            except Exception:  # noqa: BLE001
                self.door = False
                return
        if not door:
            return
        topic = str(m.get("topic") or "")[:300]

        async def judge():
            sys_ = TOPIC_SYS
            user = f"Board topic: {topic}\nThe listener just said: {text}"
            try:
                ev = await door.complete(model="haiku", system=sys_, user=user, max_s=6.0, tag="teach_topic")
                ans = str((ev or {}).get("text") or "").strip().upper()
            except Exception:  # noqa: BLE001
                return
            if ans.startswith("STOP"):
                self.n["aborts"] += 1
                self.send({"type": "teach_abort", "why": "the user moved on: " + text[:60]})
        asyncio.get_event_loop().create_task(judge())

    def stats(self) -> dict:
        return dict(self.n, first_item_ms=self.first_item_ms[-10:], on_board=len(self.items))


def summarize(items: list) -> str:
    """'the board shows: m1 (math) "3x + 2 = 11" top left; d1 (diagram) ... right' — compact and true."""
    if not items:
        return "the board is empty"
    out = []
    for it in items[:14]:
        b = it.get("bbox") or [0, 0, 0, 0]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        where = ("top" if cy < 0.34 else "bottom" if cy > 0.66 else "middle") + " " + ("left" if cx < 0.34 else "right" if cx > 0.66 else "centre")
        txt = str(it.get("text") or "").strip()
        out.append(f'{it.get("id")} ({it.get("kind")})' + (f' "{txt[:40]}"' if txt else "") + " " + where)
    return "the board shows: " + "; ".join(out)


def _jsonable(it):
    try:
        return json.loads(json.dumps(it, default=lambda o: list(o) if isinstance(o, (tuple, set)) else str(o)))
    except Exception:  # noqa: BLE001
        return {"id": str(it.get("id")), "kind": it.get("kind"), "strokes": []}
