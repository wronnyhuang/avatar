"""Render pool: N worker processes, each owning one backend instance, frames handed out round-robin to whichever is idle.

    pool = RenderPool("v2", "orb", 640, 360, procs=4, opts={"backend": "cpu", "threads": 16})
    warm = pool.start(warmup=4)                 # spawns, builds the backends, renders warm-up frames on all at once
    pool.submit(wid, job_id, index, params)     # only to an idle worker (pool.idle())
    for wid, job_id, frame, t0, t1 in pool.poll(timeout): ...

A worker holds exactly one job at a time, so a frame never queues behind another inside a worker: its latency is one
render, and N workers give ~N x the throughput.  Frames come back through one shared-memory slot per worker (no pickling
of pixels); the parent copies the slot out before it marks the worker idle again, so one slot is enough.

Every time stamp is time.monotonic() (CLOCK_MONOTONIC), the same clock in every process.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing import connection, shared_memory

import numpy as np


def _worker(wid, backend, character, w, h, opts, job_r, res_w, shm_name, cpus, warmup, warm_params=None):
    try:
        if cpus:
            os.sched_setaffinity(0, cpus)
        from live.backend_api import get_backend, REGISTRY
        cls = get_backend(backend) if backend in REGISTRY else _load(backend)
        shm = shared_memory.SharedMemory(name=shm_name)
        out = np.ndarray((h, w, 3), np.uint8, buffer=shm.buf)
        t = time.monotonic()
        be = cls(character, w, h, **opts)
        init_s = time.monotonic() - t
        from anim.schema import NPARAMS
        rows = list(warm_params) if warm_params is not None and len(warm_params) else [np.zeros(NPARAMS, np.float32)]
        times = []
        for k in range(warmup):         # real rows (speech) when given: a rest pose can be cheaper than a talking one
            t0 = time.monotonic()
            be.render(k, rows[k % len(rows)])
            times.append(time.monotonic() - t0)
        res_w.send(("ready", wid, init_s, times, os.getpid()))
        while True:
            msg = job_r.recv()
            if msg is None:
                break
            job_id, index, params = msg
            t0 = time.monotonic()
            img = be.render(index, params)
            t1 = time.monotonic()
            out[...] = img
            res_w.send(("done", wid, job_id, t0, t1, time.monotonic()))
        be.close()
        del out
        shm.close()
    except (EOFError, KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception:
        try:
            res_w.send(("error", wid, traceback.format_exc()))
        except Exception:
            pass


def _load(spec):
    """'package.module:Class' -- any class with the backend contract, for tests and new backends."""
    import importlib
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def cpu_groups(n):
    """Split this process's CPUs into n contiguous groups (for --pin)."""
    cpus = sorted(os.sched_getaffinity(0))
    k = len(cpus) // n
    return [set(cpus[i * k:(i + 1) * k]) for i in range(n)]


class RenderPool:
    def __init__(self, backend, character, w, h, procs=1, opts=None, pin=False):
        self.backend, self.character, self.w, self.h = backend, character, w, h
        self.n = int(procs)
        self.opts = dict(opts or {})
        self.pin = pin
        self.procs, self.jobs, self.res, self.shm, self.views = [], [], [], [], []
        self.busy = [None] * self.n          # job id held by each worker, None = idle
        self.pids = [None] * self.n
        self._rr = 0

    def start(self, warmup=4, timeout=600.0, warm_params=None):
        ctx = mp.get_context("spawn")      # GL contexts and fork do not mix
        groups = cpu_groups(self.n) if self.pin else [None] * self.n
        nbytes = self.w * self.h * 3
        for wid in range(self.n):
            shm = shared_memory.SharedMemory(create=True, size=nbytes)
            job_r, job_w = ctx.Pipe(duplex=False)
            res_r, res_w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_worker, name=f"render-{wid}", daemon=False,   # a backend may spawn helpers (strips)
                            args=(wid, self.backend, self.character, self.w, self.h, self.opts, job_r, res_w,
                                  shm.name, groups[wid], warmup, warm_params))
            p.start()
            job_r.close(); res_w.close()
            self.procs.append(p); self.jobs.append(job_w); self.res.append(res_r); self.shm.append(shm)
            self.views.append(np.ndarray((self.h, self.w, 3), np.uint8, buffer=shm.buf))
        warm = {}
        deadline = time.monotonic() + timeout
        while len(warm) < self.n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("render workers did not come up")
            for c in connection.wait(self.res, timeout=left):
                msg = c.recv()
                if msg[0] == "error":
                    raise RuntimeError(f"render worker {msg[1]} failed:\n{msg[2]}")
                if msg[0] == "ready":
                    _, wid, init_s, times, pid = msg
                    warm[wid] = {"init_s": init_s, "warmup_ms": [1000 * t for t in times]}
                    self.pids[wid] = pid
        return warm

    def idle(self):
        return [i for i in range(self.n) if self.busy[i] is None]

    def next_idle(self):
        """Round-robin over the idle workers, so the load (and any per-process cache) spreads evenly."""
        for k in range(self.n):
            i = (self._rr + k) % self.n
            if self.busy[i] is None:
                self._rr = (i + 1) % self.n
                return i
        return None

    def submit(self, wid, job_id, index, params):
        assert self.busy[wid] is None
        self.busy[wid] = job_id
        self.jobs[wid].send((job_id, int(index), np.asarray(params, np.float32)))

    def poll(self, timeout=None):
        """Finished frames: [(wid, job_id, frame copy, render start, render end, sent)].  Frees the workers."""
        out = []
        for c in connection.wait(self.res, timeout=timeout):
            try:
                msg = c.recv()
            except EOFError:
                raise RuntimeError("a render worker died")
            if msg[0] == "error":
                raise RuntimeError(f"render worker {msg[1]} failed:\n{msg[2]}")
            _, wid, job_id, t0, t1, ts = msg
            frame = self.views[wid].copy()
            self.busy[wid] = None
            out.append((wid, job_id, frame, t0, t1, ts))
        return out

    def close(self):
        for j in self.jobs:
            try:
                j.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()       # our own child, by its own handle
                p.join(timeout=5)
        self.views = []         # drop the numpy views first: an exported buffer cannot be closed
        for s in self.shm:
            try:
                s.close(); s.unlink()
            except Exception:
                pass
