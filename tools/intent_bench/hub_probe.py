"""End-to-end probe of the server path: open a direct-voice session on a running avatar server, stream TTS command clips as
'M' chunks in real time (140 ms exchanges, like the page), and time each {"type":"intent"} that comes back.

usage: python3 hub_probe.py <base_url> <key> <clips manifest.json> [n] [x-requested-with]
  (base_url e.g. http://127.0.0.1:10250, key = the server's state/secret; the manifest comes from gen_tts.py)
Latency = arrival of the intent at this client - end of speech in the clip (both on this client's clock), so it
includes the loopback HTTP legs + batching + ASR + matcher, not the internet path."""
import json, os, random, struct, sys, threading, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import load_pcm, speech_end  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
from check_cases import cases, summ  # noqa: E402

BASE, KEY = sys.argv[1].rstrip("/"), sys.argv[2]
HDR = {"x-requested-with": sys.argv[5] if len(sys.argv) > 5 else "avatar", "content-type": "application/json"}


def post(path, body, sid=None, raw=False):
    q = f"?k={KEY}" + (f"&sid={sid}" if sid else "")
    req = urllib.request.Request(BASE + "/" + path + q, data=body if raw else json.dumps(body).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read()


def main():
    man = json.load(open(sys.argv[3]))
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    exp = dict(cases())
    pos = [c for c in man if c["kind"] == "pos"]
    random.Random(7).shuffle(pos)
    j = json.loads(post("api/session", {"character": "fox", "quality": "360p", "codec": "jpeg", "voice": "direct"}))
    sid = j["sid"]
    print("session", sid[:6], "intent_mic", j["config"].get("intent_mic"))
    seq, after = 0, 0
    got = []
    lock = threading.Lock()
    marks = {}
    stop = threading.Event()
    upq = []

    def exchanger():
        nonlocal after, seq
        while not stop.is_set():
            with lock:
                msgs, upq[:] = list(upq), []
            body = b"".join(struct.pack("<I", len(m)) + m for m in msgs)
            t0 = time.monotonic()
            try:
                data = post("api/exchange", body, sid, raw=True) if False else None
                q = f"?k={KEY}&sid={sid}&after={after}"
                req = urllib.request.Request(BASE + "/api/exchange" + q, data=body, headers={"x-requested-with": HDR["x-requested-with"]})
                with urllib.request.urlopen(req, timeout=10) as r:
                    data = r.read()
            except Exception as e:  # noqa: BLE001
                print("exchange error", e)
                time.sleep(0.2)
                continue
            off = 0
            now = time.monotonic()
            while off + 8 <= len(data):
                s, ln = struct.unpack_from("<II", data, off)
                off += 8
                m = data[off:off + ln]
                off += ln
                after = max(after, s)
                if m[:1] == b"J":
                    o = json.loads(m[1:])
                    if o.get("type") == "intent":
                        got.append((now, o))
            dt = time.monotonic() - t0
            time.sleep(max(0, 0.143 - dt))

    th = threading.Thread(target=exchanger, daemon=True)
    th.start()
    time.sleep(2.0)
    t_play = time.monotonic()
    pcm_all = []
    sched = []
    for c in pos[:n]:
        x = load_pcm(c["file"])
        lead = np.zeros(int(0.5 * 24000), np.float32)
        tail = np.zeros(int(1.5 * 24000), np.float32)
        start = sum(len(p) for p in pcm_all) / 24000 + 0.5
        sched.append((c["text"], start + speech_end(x), exp.get(c["text"], []), start, start + len(x) / 24000 + 1.5))
        pcm_all += [lead, x, tail]
    pcm = np.concatenate(pcm_all).astype("<i2").tobytes()
    t0 = time.monotonic()
    for i in range(0, len(pcm), 960):   # 20 ms 'M' messages in real time
        target = t0 + i / 2 / 24000
        while time.monotonic() < target:
            time.sleep(0.002)
        m = struct.pack("<B3xI", 0x4D, seq) + pcm[i:i + 960]
        seq += 1
        with lock:
            upq.append(m)
    time.sleep(2.5)
    stop.set()
    th.join(2)
    post("api/bye", {}, sid)
    lats, ok, wrong = [], 0, 0
    for text, tend, e, st, nx in sched:
        abs_end = t0 + tend
        hs = [(t, o) for t, o in got if t0 + st <= t < t0 + nx + 0.5]
        ss = [summ(o["cmd"]) for _, o in hs]
        if hs and set(e) <= set(ss) and all(x in e for x in ss):
            ok += 1
            lats.append(min(t for t, _ in hs) - abs_end)
        elif any(x not in e for x in ss):
            wrong += 1
        print(f"{text!r:45} want {e} got {ss} lat {round((min(t for t, _ in hs) - abs_end) * 1000) if hs else None} ms"
              f" partial={[o['partial'] for _, o in hs]}")
    if lats:
        print(f"\n{ok}/{len(sched)} correct, {wrong} wrong; end-to-end (loopback) word-end -> intent at client: "
              f"p50 {np.percentile(lats, 50) * 1000:.0f} ms, p95 {np.percentile(lats, 95) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
