#!/usr/bin/env python3
"""Public-route overhead (tunnel / reverse proxy): GET-only probe of a service's /health through its public URL vs localhost.

    python door_probe.py --public https://<your-public-host>/health --local http://127.0.0.1:<port>/health [--n 20]

Keep-alive requests (one TLS connection, like a phone's HTTP/2 session) and fresh-connection requests (TCP+TLS
each), spaced 0.5 s, n <= 60 in total. Reports p50/p90/max ms. Nothing but GET /health is ever sent.
"""
import argparse
import http.client
import json
import ssl
import statistics
import time
import urllib.parse


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return None
    return {"n": len(xs), "p50": round(statistics.median(xs), 1), "p90": round(xs[int(0.9 * (len(xs) - 1))], 1),
            "min": round(xs[0], 1), "max": round(xs[-1], 1)}


def conn_for(u):
    p = urllib.parse.urlsplit(u)
    if p.scheme == "https":
        return http.client.HTTPSConnection(p.hostname, p.port or 443, timeout=10, context=ssl.create_default_context())
    return http.client.HTTPConnection(p.hostname, p.port or 80, timeout=10)


def probe(u, n, keep, gap=0.5):
    p = urllib.parse.urlsplit(u)
    path = p.path + (("?" + p.query) if p.query else "")
    out, codes = [], {}
    c = conn_for(u) if keep else None
    if keep:                      # warm the connection (TLS handshake) outside the timing
        c.request("GET", path); c.getresponse().read()
    for _ in range(n):
        if not keep:
            c = conn_for(u)
        t = time.perf_counter()
        c.request("GET", path, headers={"cache-control": "no-cache"})
        r = c.getresponse(); r.read()
        out.append((time.perf_counter() - t) * 1000)
        codes[r.status] = codes.get(r.status, 0) + 1
        if not keep:
            c.close()
        time.sleep(gap)
    if keep:
        c.close()
    return {"ms": stats(out), "status": codes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    n = min(a.n, 20)
    res = {"local_keepalive": probe(a.local, n, True), "public_keepalive": probe(a.public, n, True),
           "public_fresh_conn": probe(a.public, max(5, n // 2), False)}
    print(json.dumps(res, indent=1))
