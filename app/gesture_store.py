"""perception — the server's store of the gestures the user taught (app/gestures.js models), one JSON file per name.

    AVATAR_HOME/state/gestures/<slug>.json      {name, kind, examples, counters, thr, base, ... (the page's model)}

Page -> hub:  {type:'gesture_save', name, model}  {type:'gesture_forget', name}  {type:'gesture_list'}
Hub -> page:  {type:'gestures', items:[model, ...]}   (the answer to gesture_list, sent when perception starts);
              {type:'gesture_saved', name, ok:false, why} only when a save is refused
Caps: a model ≤ 1.5 MB of JSON (examples keep their frames), ≤ 60 gestures; names are free text ≤ 60 chars, the file
name is a slug + a short hash so "Peace sign" and "peace-sign" never collide and nothing escapes the folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

MAX_BYTES = 1_500_000
MAX_GESTURES = 60


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "gesture"
    return f"{s}-{hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]}"


class GestureStore:
    def __init__(self, state_dir: str, log=print):
        self.dir = os.path.join(state_dir, "gestures")
        self.log = log or (lambda *_: None)

    def _path(self, name: str) -> str:
        return os.path.join(self.dir, slug(name) + ".json")

    def items(self) -> list:
        out = []
        if not os.path.isdir(self.dir):
            return out
        for fn in sorted(os.listdir(self.dir)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.dir, fn), encoding="utf-8") as f:
                    m = json.load(f)
                if isinstance(m, dict) and isinstance(m.get("name"), str):
                    out.append(m)
            except (OSError, ValueError):
                self.log(f"gestures: unreadable {fn}")
        return out

    def save(self, name, model) -> dict:
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            return {"ok": False, "why": "bad name"}
        if not isinstance(model, dict) or model.get("name") != name or model.get("kind") not in ("static", "moving"):
            return {"ok": False, "why": "bad model"}
        data = json.dumps(model, separators=(",", ":"))
        if len(data) > MAX_BYTES:
            return {"ok": False, "why": f"model too big ({len(data)} bytes)"}
        p = self._path(name)
        if not os.path.exists(p) and len(self.items()) >= MAX_GESTURES:
            return {"ok": False, "why": "too many gestures"}
        os.makedirs(self.dir, exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, p)
        self.log(f"gestures: saved {name!r} ({model.get('kind')}, {len(model.get('examples') or [])} examples, {len(data)} B)")
        return {"ok": True}

    def forget(self, name) -> dict:
        if name == "all":
            n = 0
            for m in self.items():
                try:
                    os.remove(self._path(m["name"]))
                    n += 1
                except OSError:
                    pass
            return {"ok": True, "removed": n}
        if not isinstance(name, str):
            return {"ok": False, "why": "bad name"}
        try:
            os.remove(self._path(name))
            return {"ok": True}
        except OSError:
            return {"ok": False, "why": "no such gesture"}

    def handle(self, m: dict) -> dict | None:
        """One page message -> the reply to send back (or None)."""
        t = m.get("type")
        if t == "gesture_save":
            r = self.save(m.get("name"), m.get("model"))
            return None if r["ok"] else {"type": "gesture_saved", "name": m.get("name"), **r}
        if t == "gesture_forget":
            self.forget(m.get("name"))
            return None
        if t == "gesture_list":
            return {"type": "gestures", "items": self.items(), "t": round(time.time(), 1)}
        return None
