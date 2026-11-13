"""recstore.py — avatar session recordings on the server: chunk spool, assembler (ffmpeg -> mp4), central index.

Stdlib only. Used by the avatar app (recorder_hub.py mounts it as HTTP endpoints); a standalone sweeper can use the
same module to finish sessions whose app went away.

The phone records (app/recorder.js: MediaRecorder on a canvas composite of the stage + a WebAudio mix of the user's mic and the
avatar's voice) and uploads a CHUNK every few seconds, numbered 0, 1, 2 … (chunk 0 carries the container header).
Chunks may arrive out of order, twice, or never; the client may never say 'end' (phones die mid-session).

Layout (ROOT = $AVATAR_REC_ROOT, else <repo>/recordings/avatar-sessions — one place for every link variant):
  ROOT/.spool/<rid>/meta.json        start metadata (client + server)
  ROOT/.spool/<rid>/c000000.bin …    chunks by sequence number (tmp + rename; a re-sent seq replaces/keeps one copy)
  ROOT/.spool/<rid>/notes.jsonl      mid-session notes (character switches)
  ROOT/.spool/<rid>/end.json         the client's 'end' (n chunks made, dropped seqs, reason)
  ROOT/.work/<rid>/                  claimed for assembly (atomic rename: exactly one process assembles a rid;
                                     a flock on .work/<rid>/.lock marks the live assembler, a dead one's rid is retried)
  ROOT/.failed/<rid>/                assembly failed: chunks kept + error.txt
  ROOT/<YYYY-MM-DD>/<name>.mp4       the finished session (H.264 + AAC, faststart) …
  ROOT/<YYYY-MM-DD>/<name>.json      … and its metadata (one record of index.json)
  ROOT/index.json                    every session, newest first (rebuilt after each assembly, under a flock)

A rid is due for assembly when: its 'end' came and every expected chunk is there; or 'end' came END_GRACE_S ago; or no
chunk arrived for IDLE_S (no 'end' — the phone died / lost the network / the tab was killed). A chunk for a rid that is
no longer in the spool gets `Closed` (HTTP 410) — the page then starts a new recording PART (fresh header).
"""
from __future__ import annotations

import datetime as dt
import errno
import fcntl
import json
import os
import re
import secrets
import shutil
import subprocess
import time

RID_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9]{1,16}-[0-9a-f]{8}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
WORD_RE = re.compile(r"[^a-z0-9]+")


def default_root() -> str:
    return os.environ.get("AVATAR_REC_ROOT") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         "recordings", "avatar-sessions")


def _word(x, default: str = "", n: int = 16) -> str:
    s = WORD_RE.sub("", str(x or "").lower())[:n]
    return s or default


def _text(x, n: int = 120) -> str:
    return "".join(ch for ch in str(x or "") if ch.isprintable())[:n]


def _num(x, lo: float, hi: float):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v < lo or v > hi:
        return None
    return v


class Closed(Exception):
    """The rid is not (or no longer) accepting chunks: assembled, claimed, failed or never started."""


class Refused(Exception):
    """A request the store will not take (bad sequence, too big, disk full)."""


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}-{secrets.token_hex(3)}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _write_json(path: str, obj) -> None:
    _write_atomic(path, json.dumps(obj, indent=1, sort_keys=True).encode())


def _read_json(path: str, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class RecStore:
    CHUNK_MAX = 12 * 1024 * 1024        # one upload (3 s chunks are ~0.3-0.6 MB; keep it under a 16 MiB proxy body limit)
    SEQ_MAX = 100_000                   # ~83 h of 3 s chunks
    IDLE_S = 150.0                      # no chunk for this long and no 'end' -> assemble what is there
    END_GRACE_S = 45.0                  # after 'end', wait this long for chunks still in flight
    STALE_WORK_S = 120.0                # a .work/<rid> whose assembler died (lock free) is retried after this
    MIN_FREE_GB = 15.0                  # refuse new recordings below this much free disk

    def __init__(self, root: str | None = None, log=None, idle_s: float | None = None, end_grace_s: float | None = None,
                 min_free_gb: float | None = None, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", nice: int = 15,
                 threads: int = 2):
        self.root = os.path.abspath(os.path.expanduser(root or default_root()))
        self.log = log or (lambda m: None)
        if idle_s is not None:
            self.IDLE_S = float(idle_s)
        if end_grace_s is not None:
            self.END_GRACE_S = float(end_grace_s)
        if min_free_gb is not None:
            self.MIN_FREE_GB = float(min_free_gb)
        self.ffmpeg, self.ffprobe, self.nice, self.threads = ffmpeg, ffprobe, int(nice), int(threads)
        self.spool = os.path.join(self.root, ".spool")
        self.work = os.path.join(self.root, ".work")
        self.failed = os.path.join(self.root, ".failed")
        for d in (self.root, self.spool, self.work, self.failed):
            os.makedirs(d, exist_ok=True)
        self.stats = {"started": 0, "chunks": 0, "dup": 0, "bytes": 0, "closed": 0, "assembled": 0, "failed": 0}

    # ------------------------------------------------------------------------------------------------ the protocol
    def _dir(self, rid: str) -> str:
        if not isinstance(rid, str) or not RID_RE.match(rid):
            raise Closed("bad rid")
        return os.path.join(self.spool, rid)

    def free_gb(self) -> float:
        try:
            return shutil.disk_usage(self.root).free / 1e9
        except OSError:
            return 0.0

    def start(self, client: dict, hub: dict | None = None) -> dict:
        """A new recording (one session, or one part of one). client = what the page says about itself; hub = what
        the app knows (variant, voice default, hub session). Returns {"rid": …, "chunk_max": …}."""
        if self.free_gb() < self.MIN_FREE_GB:
            raise Refused(f"less than {self.MIN_FREE_GB:g} GB free on the recordings disk")
        c = client if isinstance(client, dict) else {}
        h = hub if isinstance(hub, dict) else {}
        variant = _word(h.get("variant") or c.get("variant"), "local")
        now = time.time()
        lt = time.localtime(now)
        rid = f"{time.strftime('%Y%m%d-%H%M%S', lt)}-{variant}-{secrets.token_hex(4)}"
        meta = {
            "rid": rid, "variant": variant, "test": bool(h.get("test")) or variant == "test" or bool(c.get("test")),
            "started_ts": round(now, 3), "started": dt.datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
            "client_ts": _num(c.get("client_ts"), 0, 1e13),
            "character": _word(c.get("character"), "unknown", 24), "character_name": _text(c.get("character_name"), 40),
            "voice": _word(c.get("voice"), "", 24), "voice_label": _text(c.get("voice_label"), 60),
            "mime": _text(c.get("mime"), 80), "width": int(_num(c.get("width"), 0, 8192) or 0),
            "height": int(_num(c.get("height"), 0, 8192) or 0), "fps": _num(c.get("fps"), 0, 120),
            "timeslice_ms": _num(c.get("timeslice_ms"), 0, 600000), "group": _word(c.get("group"), "", 24),
            "part": int(_num(c.get("part"), 1, 10000) or 1), "sid": _text(c.get("sid"), 6),
            "automated": bool(c.get("automated")), "ua": _text(c.get("ua"), 120),
            "app": {k: _text(v, 80) for k, v in h.items() if k in ("service", "home", "version", "provider")},
        }
        d = os.path.join(self.spool, rid)
        os.makedirs(d)
        _write_json(os.path.join(d, "meta.json"), meta)
        self.stats["started"] += 1
        self.log(f"rec {rid}: started ({meta['character']}, {meta['voice'] or '?'}, {meta['mime'] or '?'}, "
                 f"{meta['width']}x{meta['height']})")
        return {"rid": rid, "chunk_max": self.CHUNK_MAX}

    def put_chunk(self, rid: str, seq, data: bytes) -> dict:
        d = self._dir(rid)
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            raise Refused("bad seq")
        if seq < 0 or seq > self.SEQ_MAX:
            raise Refused("bad seq")
        if len(data) > self.CHUNK_MAX:
            raise Refused("chunk too big")
        if not os.path.isdir(d):
            self.stats["closed"] += 1
            raise Closed("closed")
        path = os.path.join(d, f"c{seq:06d}.bin")
        try:
            if os.path.getsize(path) == len(data):
                self.stats["dup"] += 1
                os.utime(d)                       # a re-send still counts as activity
                return {"ok": True, "seq": seq, "dup": True}
        except OSError:
            pass
        if seq % 50 == 0 and self.free_gb() < self.MIN_FREE_GB / 3:
            raise Refused("recordings disk is full")
        try:
            _write_atomic(path, data)
        except FileNotFoundError:               # claimed for assembly between the check and the write
            self.stats["closed"] += 1
            raise Closed("closed")
        self.stats["chunks"] += 1
        self.stats["bytes"] += len(data)
        return {"ok": True, "seq": seq}

    def have(self, d: str) -> list[int]:
        out = []
        try:
            names = os.listdir(d)
        except OSError:
            return out
        for n in names:
            if n.startswith("c") and n.endswith(".bin") and n[1:-4].isdigit():
                out.append(int(n[1:-4]))
        return sorted(out)

    def status(self, rid: str) -> dict:
        d = self._dir(rid)
        if not os.path.isdir(d):
            raise Closed("closed")
        have = self.have(d)
        return {"ok": True, "rid": rid, "have": have, "next": (have[-1] + 1) if have else 0,
                "ended": os.path.isfile(os.path.join(d, "end.json"))}

    def note(self, rid: str, note: dict) -> dict:
        d = self._dir(rid)
        if not os.path.isdir(d):
            raise Closed("closed")
        n = note if isinstance(note, dict) else {}
        rec = {"ts": round(time.time(), 3), "t_s": _num(n.get("t_s"), 0, 1e7)}
        if n.get("character"):
            rec["character"] = _word(n.get("character"), "", 24)
        if n.get("voice"):
            rec["voice"] = _word(n.get("voice"), "", 24)
        if n.get("text"):
            rec["text"] = _text(n.get("text"), 200)
        with open(os.path.join(d, "notes.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        os.utime(d)
        return {"ok": True}

    def end(self, rid: str, info: dict) -> dict:
        d = self._dir(rid)
        if not os.path.isdir(d):
            raise Closed("closed")
        i = info if isinstance(info, dict) else {}
        dropped = i.get("dropped") if isinstance(i.get("dropped"), list) else []
        rec = {"ts": round(time.time(), 3), "n": int(_num(i.get("n"), 0, self.SEQ_MAX + 1) or 0),
               "dropped": sorted({int(x) for x in dropped[:5000] if _num(x, 0, self.SEQ_MAX) is not None}),
               "duration_ms": _num(i.get("duration_ms"), 0, 1e9), "reason": _word(i.get("reason"), "end", 24),
               "client": {k: v for k, v in (i.get("stats") or {}).items()
                          if isinstance(k, str) and len(k) < 32 and isinstance(v, (int, float)) and not isinstance(v, bool)}
               if isinstance(i.get("stats"), dict) else {}}
        _write_json(os.path.join(d, "end.json"), rec)
        os.utime(d)
        self.log(f"rec {rid}: end ({rec['reason']}, n={rec['n']}, have {len(self.have(d))})")
        return {"ok": True, "complete": self._complete(d, rec)}

    def _complete(self, d: str, end: dict) -> bool:
        have = set(self.have(d))
        want = set(range(int(end.get("n") or 0))) - set(end.get("dropped") or [])
        return bool(want) and want <= have

    # ------------------------------------------------------------------------------------------------ assembly
    def due(self, now: float | None = None) -> list[tuple[str, str]]:
        """[(rid, why)] ready for assembly: 'end' (complete), 'end-partial' (grace over), 'idle' (no 'end'),
        'recover' (a dead assembler's claim)."""
        now = time.time() if now is None else now
        out = []
        try:
            rids = sorted(os.listdir(self.spool))
        except OSError:
            rids = []
        for rid in rids:
            if not RID_RE.match(rid):
                continue
            d = os.path.join(self.spool, rid)
            try:
                mt = os.stat(d).st_mtime
            except OSError:
                continue
            end = _read_json(os.path.join(d, "end.json"))
            if isinstance(end, dict):
                if self._complete(d, end):
                    out.append((rid, "end"))
                elif now - float(end.get("ts") or mt) > self.END_GRACE_S:
                    out.append((rid, "end-partial"))
            elif now - mt > self.IDLE_S:
                out.append((rid, "idle"))
        try:
            wrids = sorted(os.listdir(self.work))
        except OSError:
            wrids = []
        for rid in wrids:
            w = os.path.join(self.work, rid)
            if ".done-" in rid:                      # a tombstone whose removal was interrupted
                try:
                    if now - os.stat(w).st_mtime > 3600:
                        shutil.rmtree(w, ignore_errors=True)
                except OSError:
                    pass
                continue
            if RID_RE.match(rid) and os.path.isdir(w):
                try:
                    if now - os.stat(w).st_mtime > self.STALE_WORK_S:
                        out.append((rid, "recover"))
                except OSError:
                    pass
        return out

    def _lock(self, w: str):
        try:
            fd = os.open(os.path.join(w, ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
        except OSError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return None
        return fd

    def claim(self, rid: str, why: str = ""):
        """Move the rid out of the spool (no more chunks) and lock it. Returns (work dir, lock fd) or None."""
        if not RID_RE.match(rid or ""):
            return None
        w = os.path.join(self.work, rid)
        if why != "recover":
            try:
                os.rename(os.path.join(self.spool, rid), w)
            except OSError as e:
                if e.errno in (errno.ENOENT, errno.ENOTEMPTY, errno.EEXIST):
                    return None
                raise
        elif not os.path.isdir(w):
            return None
        fd = self._lock(w)
        if fd is None:
            return None
        if not os.path.isdir(w):                  # the previous owner finished between our listdir and our lock
            os.close(fd)
            return None
        os.utime(w)
        return w, fd

    def assemble(self, rid: str, why: str = "") -> dict | None:
        """Claim + assemble one rid (blocking: ffmpeg runs here — call it from a thread). Returns the session record,
        {"empty": True} for a recording with no chunks, or None if another process has it."""
        c = self.claim(rid, why)
        if c is None:
            return None
        w, fd = c
        try:
            return self._assemble(rid, w, why)
        except Exception as e:  # noqa: BLE001
            self.stats["failed"] += 1
            self.log(f"rec {rid}: assembly FAILED ({type(e).__name__}: {str(e)[:300]})")
            try:
                with open(os.path.join(w, "error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {why}: {type(e).__name__}: {e}\n")
                dst = os.path.join(self.failed, rid)
                if os.path.exists(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                os.rename(w, dst)
            except OSError:
                pass
            return {"rid": rid, "failed": True, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        finally:
            os.close(fd)

    def _discard(self, w: str) -> None:
        """Remove a finished work dir: renamed first (a name due() ignores), so no sweeper can 'recover' it while the
        lock file is already gone and the chunks are half deleted."""
        tomb = f"{w}.done-{os.getpid()}-{secrets.token_hex(2)}"
        try:
            os.rename(w, tomb)
        except OSError:
            tomb = w
        shutil.rmtree(tomb, ignore_errors=True)

    def _assemble(self, rid: str, w: str, why: str) -> dict:
        t0 = time.time()
        meta = _read_json(os.path.join(w, "meta.json"), {}) or {}
        end = _read_json(os.path.join(w, "end.json"))
        notes = []
        try:
            with open(os.path.join(w, "notes.jsonl"), encoding="utf-8") as f:
                for line in f:
                    try:
                        notes.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
        have = self.have(w)
        if not have:
            self.log(f"rec {rid}: no chunks ({why}); nothing to keep")
            self._discard(w)
            return {"rid": rid, "empty": True}
        n = int(end.get("n") or 0) if isinstance(end, dict) else 0
        n = max(n, have[-1] + 1)
        dropped = list(end.get("dropped") or []) if isinstance(end, dict) else []
        hs = set(have)
        missing = [s for s in range(n) if s not in hs]
        mime = str(meta.get("mime") or "")
        ext = "mp4" if "mp4" in mime else "webm"
        raw = os.path.join(w, f"raw.{ext}")
        raw_bytes = 0
        with open(raw, "wb") as out:
            for s in have:
                with open(os.path.join(w, f"c{s:06d}.bin"), "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 20)
            raw_bytes = out.tell()
        started = float(meta.get("started_ts") or os.stat(w).st_mtime)
        lt = time.localtime(started)
        date = time.strftime("%Y-%m-%d", lt)
        rand = rid.rsplit("-", 1)[-1]
        variant = _word(meta.get("variant"), "local")
        character = _word(meta.get("character"), "unknown", 24)
        name = f"{time.strftime('%H%M%S', lt)}-{variant}-{character}-{rand}"
        if int(meta.get("part") or 1) > 1:
            name += f"-p{int(meta['part'])}"
        ddir = os.path.join(self.root, date)
        os.makedirs(ddir, exist_ok=True)
        mp4 = os.path.join(ddir, name + ".mp4")
        part = mp4 + ".part"
        self._transcode(raw, part, raw_bytes)
        probe = self.probe(part)
        if not probe.get("video") and not probe.get("audio"):
            raise RuntimeError("ffmpeg produced no audio or video stream")
        os.replace(part, mp4)
        chars = [{"t_s": 0.0, "character": character}]
        for nt in notes:
            if nt.get("character") and nt["character"] != chars[-1]["character"]:
                chars.append({"t_s": nt.get("t_s"), "character": nt["character"]})
        rec = {k: meta.get(k) for k in ("rid", "variant", "test", "started", "started_ts", "client_ts", "character",
                                         "character_name", "voice", "voice_label", "mime", "group", "part", "sid",
                                         "automated", "ua", "app", "timeslice_ms")}
        rec.update({
            "name": name, "date": date, "file": f"{date}/{name}.mp4", "characters": chars,
            "duration_s": probe.get("duration"), "size_bytes": os.path.getsize(mp4), "raw_bytes": raw_bytes,
            "width": probe.get("width"), "height": probe.get("height"), "fps": probe.get("fps"),
            "video": probe.get("video"), "audio": probe.get("audio"),
            "chunks": len(have), "chunks_expected": n, "missing": missing[:500], "dropped": dropped[:500],
            "end": (end.get("reason") if isinstance(end, dict) else None) or "none", "assembled_by": why,
            "ended_ts": round(float(end.get("ts")), 3) if isinstance(end, dict) and end.get("ts") else
            round(max(os.stat(os.path.join(w, f"c{s:06d}.bin")).st_mtime for s in have[-1:]), 3),
            "client": (end.get("client") if isinstance(end, dict) else None) or {},
            "assembled_ts": round(time.time(), 3), "assemble_s": round(time.time() - t0, 2),
        })
        _write_json(os.path.join(ddir, name + ".json"), rec)
        self._discard(w)
        self.stats["assembled"] += 1
        self.log(f"rec {rid}: assembled {rec['file']} ({rec['duration_s']} s, {rec['size_bytes'] // 1024} KB, "
                 f"{len(have)}/{n} chunks, {why}, {rec['assemble_s']} s)")
        self.rebuild_index()
        return rec

    def _transcode(self, src: str, dst: str, nbytes: int) -> None:
        """Any MediaRecorder output (webm VP8/VP9/H.264 + Opus, Safari's fragmented mp4) -> H.264 + AAC mp4 that
        plays everywhere. Corrupt / missing stretches are skipped; audio gaps are filled with silence so A/V stays
        in sync; variable frame timing is kept (a gap shows as a held frame)."""
        cmd = ["nice", "-n", str(self.nice), self.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-fflags", "+genpts+discardcorrupt", "-err_detect", "ignore_err", "-i", src,
               "-map", "0:v:0?", "-map", "0:a:0?",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-fps_mode", "vfr", "-threads", str(self.threads),
               "-c:a", "aac", "-b:a", "96k", "-af", "aresample=async=1:first_pts=0",
               "-movflags", "+faststart", "-f", "mp4", dst]
        timeout = 600 + nbytes / 20_000           # generous: ~1 s per 20 KB (~6 s of video) on top of 10 min
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if r.returncode != 0 or not os.path.isfile(dst) or os.path.getsize(dst) < 1024:
            err = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-6:]
            raise RuntimeError(f"ffmpeg exit {r.returncode}: " + " | ".join(err))

    def probe(self, path: str) -> dict:
        r = subprocess.run([self.ffprobe, "-v", "error", "-show_entries",
                            "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,duration,sample_rate",
                            "-of", "json", path], capture_output=True, timeout=120)
        try:
            j = json.loads(r.stdout or b"{}")
        except ValueError:
            j = {}
        out: dict = {"duration": None, "video": None, "audio": None, "width": None, "height": None, "fps": None}
        try:
            out["duration"] = round(float(j.get("format", {}).get("duration")), 2)
        except (TypeError, ValueError):
            pass
        for s in j.get("streams") or []:
            if s.get("codec_type") == "video" and not out["video"]:
                out["video"] = s.get("codec_name")
                out["width"], out["height"] = s.get("width"), s.get("height")
                try:
                    a, b = str(s.get("avg_frame_rate") or "0/1").split("/")
                    out["fps"] = round(float(a) / float(b), 2) if float(b) else None
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
                out["video_duration"] = _num(s.get("duration"), 0, 1e7)
            elif s.get("codec_type") == "audio" and not out["audio"]:
                out["audio"] = s.get("codec_name")
                out["audio_duration"] = _num(s.get("duration"), 0, 1e7)
        return out

    # ------------------------------------------------------------------------------------------------ the index
    def sessions(self) -> list[dict]:
        out = []
        try:
            dates = sorted((x for x in os.listdir(self.root) if DATE_RE.match(x)), reverse=True)
        except OSError:
            dates = []
        for date in dates:
            dd = os.path.join(self.root, date)
            try:
                names = os.listdir(dd)
            except OSError:
                continue
            for n in names:
                if not n.endswith(".json"):
                    continue
                rec = _read_json(os.path.join(dd, n))
                if isinstance(rec, dict) and rec.get("file") and os.path.isfile(os.path.join(self.root, rec["file"])):
                    out.append(rec)
        out.sort(key=lambda r: float(r.get("started_ts") or 0), reverse=True)
        return out

    def rebuild_index(self) -> dict:
        fd = os.open(os.path.join(self.root, ".index.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            sess = self.sessions()
            idx = {"updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "count": len(sess),
                   "total_bytes": sum(int(s.get("size_bytes") or 0) for s in sess),
                   "total_s": round(sum(float(s.get("duration_s") or 0) for s in sess), 1), "sessions": sess}
            _write_json(os.path.join(self.root, "index.json"), idx)
            return idx
        finally:
            os.close(fd)

    def pending(self) -> list[dict]:
        """Recordings still uploading / waiting (for the index page)."""
        out = []
        for base, state in ((self.spool, "recording"), (self.work, "assembling")):
            try:
                rids = os.listdir(base)
            except OSError:
                continue
            for rid in rids:
                if not RID_RE.match(rid):
                    continue
                d = os.path.join(base, rid)
                meta = _read_json(os.path.join(d, "meta.json"), {}) or {}
                have = self.have(d)
                try:
                    nbytes = sum(os.path.getsize(os.path.join(d, f"c{s:06d}.bin")) for s in have)
                    mt = os.stat(d).st_mtime
                except OSError:
                    continue
                out.append({"rid": rid, "state": state, "variant": meta.get("variant"), "character": meta.get("character"),
                            "started": meta.get("started"), "started_ts": meta.get("started_ts"), "chunks": len(have),
                            "bytes": nbytes, "idle_s": round(time.time() - mt, 1), "test": meta.get("test"),
                            "ended": os.path.isfile(os.path.join(d, "end.json"))})
        out.sort(key=lambda r: float(r.get("started_ts") or 0), reverse=True)
        return out

    def file_path(self, date: str, name: str) -> str | None:
        """The on-disk path of ROOT/<date>/<name> (an .mp4 of a finished session) or None — no traversal."""
        if not DATE_RE.match(date or "") or not NAME_RE.match(name or "") or not name.endswith(".mp4"):
            return None
        p = os.path.join(self.root, date, name)
        return p if os.path.isfile(p) else None


async def sweep_forever(store: RecStore, interval: float = 10.0, kick=None) -> None:
    """Assemble whatever is due, one at a time, in a worker thread (asyncio). `kick` (asyncio.Event) wakes it early."""
    import asyncio
    while True:
        try:
            for rid, why in store.due():
                await asyncio.to_thread(store.assemble, rid, why)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            store.log(f"rec sweep: {type(e).__name__}: {str(e)[:200]}")
        try:
            if kick is not None:
                try:
                    await asyncio.wait_for(kick.wait(), interval)
                except asyncio.TimeoutError:
                    pass
                kick.clear()
            else:
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise
