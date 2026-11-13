"""recorder_hub.py — the avatar app's recording endpoints. Drop-in: one call in make_app().

    import recorder_hub
    recorder_hub.mount(app, authed=authed, home=HOME, cfg=hub.cfg, log=log)

The page's app/recorder.js records the stage (avatar video + the user's self-view) and both voices and uploads chunks here;
recstore.py keeps them in ONE place (<repo>/recordings/avatar-sessions by default, every link variant) and turns each
session into an mp4 + index record. Endpoints (all need ?k=<page secret>; all but /end also the x-requested-with header,
because the page's pagehide 'end' goes out as a sendBeacon, which cannot set headers):

  POST /api/rec/start   {character, character_name, voice, voice_label, mime, width, height, fps, timeslice_ms, group,
                         part, sid, client_ts, automated, ua}            -> {ok, rid, chunk_max}   (503 = not recording)
  POST /api/rec/chunk?rid=&seq=     body = the chunk bytes               -> {ok, seq, dup?}  (410 = rid closed -> new part)
  GET  /api/rec/status?rid=                                              -> {ok, have:[seq…], next, ended}
  POST /api/rec/note?rid=  {t_s, character?, voice?, text?}              -> {ok}
  POST /api/rec/end?rid=   {n, dropped:[seq…], duration_ms, reason, stats} (JSON body or query n/reason) -> {ok}

Config (state/config.json "recording", all optional): {"enabled": true, "root": "recordings/avatar-sessions",
"variant": "dev", "idle_s": 150, "end_grace_s": 45, "min_free_gb": 15}. The link variant is env AVATAR_VARIANT, else
recording.variant, else the AVATAR_HOME folder name without "avatar-" (avatar-dev -> dev,
avatar-live -> live). No vendor is involved: chunks and mp4s stay on the server.
"""
from __future__ import annotations

import asyncio
import json
import os

from aiohttp import web

import recstore

ROUTE_PREFIX = "/api/rec/"


def variant_of(home: str, rc: dict) -> str:
    v = os.environ.get("AVATAR_VARIANT") or rc.get("variant")
    if not v:
        base = os.path.basename(os.path.normpath(home or ""))
        v = base[len("avatar-"):] if base.startswith("avatar-") else (base or "local")
    return recstore._word(v, "local")


def mount(app: web.Application, *, authed, home: str, cfg: dict, log=print, sweep: bool = True) -> recstore.RecStore | None:
    rc = cfg.get("recording") if isinstance(cfg.get("recording"), dict) else {}
    enabled = cfg.get("recording") is not False and rc.get("enabled", True) is not False
    variant = variant_of(home, rc)
    store = None
    if enabled:
        try:
            store = recstore.RecStore(root=rc.get("root") or None, log=log, idle_s=rc.get("idle_s"),
                                      end_grace_s=rc.get("end_grace_s"), min_free_gb=rc.get("min_free_gb"))
        except OSError as e:
            log(f"recording off: {e}")
            store = None
    hubinfo = {"variant": variant, "test": variant == "test", "home": os.path.basename(os.path.normpath(home or "")),
               "provider": str(cfg.get("provider") or "")}
    kick = asyncio.Event()

    def deny() -> web.Response:
        return web.json_response({"ok": False, "error": "forbidden"}, status=403)

    def nocache(r: web.StreamResponse) -> web.StreamResponse:
        r.headers["Cache-Control"] = "no-store"
        return r

    def err(e: Exception) -> web.Response:
        if isinstance(e, recstore.Closed):
            return nocache(web.json_response({"ok": False, "error": "closed", "closed": True}, status=410))
        if isinstance(e, recstore.Refused):
            st = 413 if "big" in str(e) else 507 if "disk" in str(e) else 400
            return nocache(web.json_response({"ok": False, "error": str(e)}, status=st))
        log(f"rec endpoint error: {type(e).__name__}: {str(e)[:200]}")
        return nocache(web.json_response({"ok": False, "error": "server error"}, status=500))

    async def body_json(req: web.Request) -> dict:
        try:
            raw = await req.content.read(65536)
            d = json.loads(raw or b"{}")
            return d if isinstance(d, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    async def rec_start(req: web.Request):
        if not authed(req):
            return deny()
        if store is None:
            return nocache(web.json_response({"ok": False, "error": "recording is off here", "off": True}, status=503))
        try:
            r = store.start(await body_json(req), hubinfo)
        except Exception as e:  # noqa: BLE001
            return err(e)
        return nocache(web.json_response({"ok": True, "variant": variant, **r}))

    async def rec_chunk(req: web.Request):
        if not authed(req):
            return deny()
        if store is None:
            return err(recstore.Closed("off"))
        cap = store.CHUNK_MAX
        if req.content_length is not None and req.content_length > cap:
            return err(recstore.Refused("chunk too big"))
        buf = bytearray()
        try:
            while True:
                b = await req.content.read(1 << 16)
                if not b:
                    break
                buf += b
                if len(buf) > cap:
                    return err(recstore.Refused("chunk too big"))
        except (asyncio.TimeoutError, ConnectionError) as e:
            return nocache(web.json_response({"ok": False, "error": f"upload cut: {type(e).__name__}"}, status=400))
        try:
            r = store.put_chunk(req.query.get("rid", ""), req.query.get("seq"), bytes(buf))
        except Exception as e:  # noqa: BLE001
            return err(e)
        return nocache(web.json_response(r))

    async def rec_status(req: web.Request):
        if not authed(req):
            return deny()
        if store is None:
            return err(recstore.Closed("off"))
        try:
            return nocache(web.json_response(store.status(req.query.get("rid", ""))))
        except Exception as e:  # noqa: BLE001
            return err(e)

    async def rec_note(req: web.Request):
        if not authed(req):
            return deny()
        if store is None:
            return err(recstore.Closed("off"))
        try:
            return nocache(web.json_response(store.note(req.query.get("rid", ""), await body_json(req))))
        except Exception as e:  # noqa: BLE001
            return err(e)

    async def rec_end(req: web.Request):
        if not authed(req, api=False):           # sendBeacon at pagehide: k only (the rid is the other half)
            return deny()
        if store is None:
            return err(recstore.Closed("off"))
        d = await body_json(req)
        for k in ("n", "reason", "duration_ms"):
            if k not in d and req.query.get(k) is not None:
                d[k] = req.query.get(k)
        try:
            r = store.end(req.query.get("rid", "") or str(d.get("rid") or ""), d)
        except Exception as e:  # noqa: BLE001
            return err(e)
        kick.set()
        return nocache(web.json_response(r))

    app.router.add_post(ROUTE_PREFIX + "start", rec_start)
    app.router.add_post(ROUTE_PREFIX + "chunk", rec_chunk)
    app.router.add_get(ROUTE_PREFIX + "status", rec_status)
    app.router.add_post(ROUTE_PREFIX + "note", rec_note)
    app.router.add_post(ROUTE_PREFIX + "end", rec_end)

    if store is not None and sweep:
        async def _on_startup(app_: web.Application) -> None:
            app_["rec_sweeper"] = asyncio.create_task(recstore.sweep_forever(store, 10.0, kick))

        async def _on_cleanup(app_: web.Application) -> None:
            t = app_.get("rec_sweeper")
            if t:
                t.cancel()
                try:
                    await t
                except BaseException:  # noqa: BLE001
                    pass

        app.on_startup.append(_on_startup)
        app.on_cleanup.append(_on_cleanup)
    log(f"recording: {'on' if store else 'OFF'} (variant {variant}{', root ' + store.root if store else ''})")
    return store
