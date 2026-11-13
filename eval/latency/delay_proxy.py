#!/usr/bin/env python3
"""Emulate a phone <-> public-link (tunnel / reverse proxy) round trip in front of a local instance: every request is delayed by --up-ms (+ jitter)
before it is forwarded and every response by --down-ms (+ jitter) before it is returned, so a localhost run sees the
round trip a phone on a mobile network sees (one measured session: exchange RTT p50 ~115 ms, p95 spikes 600-900 ms).

    python delay_proxy.py --listen 18952 --target http://127.0.0.1:18950 --up-ms 55 --down-ms 55 --jitter-ms 15 [--spike-p 0.02 --spike-ms 500] [--up-kbps 600]

HTTP/1.1, no streaming (the avatar's public route is request/response only). Concurrency is unlimited, like HTTP/2.
"""
import argparse
import asyncio
import random

import aiohttp
from aiohttp import web


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--up-ms", type=float, default=55)
    ap.add_argument("--down-ms", type=float, default=55)
    ap.add_argument("--jitter-ms", type=float, default=15)
    ap.add_argument("--spike-p", type=float, default=0.0)
    ap.add_argument("--spike-ms", type=float, default=500)
    ap.add_argument("--up-kbps", type=float, default=0, help="serial uplink capacity (0 = unlimited): request bodies "
                    "queue behind each other like a phone's radio uplink")
    a = ap.parse_args()

    def d(base):
        x = base + random.uniform(0, a.jitter_ms)
        if a.spike_p and random.random() < a.spike_p:
            x += a.spike_ms
        return x / 1000

    link = {"busy": 0.0}

    async def handler(req: web.Request):
        body = await req.read()
        if a.up_kbps:
            loop = asyncio.get_running_loop()
            now = loop.time()
            start = max(now, link["busy"])
            link["busy"] = start + (len(body) + 400) * 8 / (a.up_kbps * 1000)     # + ~400 B of headers
            await asyncio.sleep(link["busy"] - now)
        await asyncio.sleep(d(a.up_ms))
        hdr = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length", "transfer-encoding")}
        async with req.app["s"].request(req.method, a.target + str(req.rel_url), headers=hdr, data=body or None,
                                        allow_redirects=False) as r:
            data = await r.read()
            out = web.Response(status=r.status, body=data)
            for k, v in r.headers.items():
                if k.lower() not in ("content-length", "transfer-encoding", "content-encoding", "connection"):
                    out.headers[k] = v
        await asyncio.sleep(d(a.down_ms))
        return out

    async def mk():
        app = web.Application(client_max_size=64 * 1024 * 1024)
        app["s"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30),
                                         connector=aiohttp.TCPConnector(limit=0, force_close=False))
        app.router.add_route("*", "/{tail:.*}", handler)
        return app
    web.run_app(mk(), host="127.0.0.1", port=a.listen, print=None)


if __name__ == "__main__":
    main()
