"""render-perf: GPU timestamps (GL_TIMESTAMP via glQueryCounter) for the mesh passes.

RTEngine._draw already wraps its passes in GL_TIME_ELAPSED queries (`prof` dict); those cannot nest, and the mesh passes
run INSIDE its "comp_dof" query.  Timestamps can: `GpuStamps.mark(name)` records a timestamp, `collect()` (after the
frame) returns {name: ms since the previous mark}.  Raw GL through ctypes (GLVND dispatches to the current context);
`ok` is False when the entry points are missing (then mark/collect are no-ops).  Zero cost when not enabled."""
import ctypes

GL_TIMESTAMP = 0x8E28
GL_QUERY_RESULT = 0x8866


class GpuStamps:
    def __init__(self, pool=64):
        self.ok = False
        self.pool = []
        self.used = []          # (name, query id)
        try:
            gl = None
            for name in ("libOpenGL.so.0", "libGL.so.1"):
                try:
                    gl = ctypes.CDLL(name); break
                except OSError:
                    continue
            if gl is None:
                return
            self.gl = gl
            gl.glGenQueries.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
            gl.glQueryCounter.argtypes = [ctypes.c_uint, ctypes.c_uint]
            gl.glGetQueryObjectui64v.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64)]
            ids = (ctypes.c_uint*pool)()
            gl.glGenQueries(pool, ids)
            self.pool = list(ids)
            self.ok = all(i != 0 for i in self.pool)
        except Exception:
            self.ok = False

    def mark(self, name):
        if not self.ok or not self.pool:
            return
        q = self.pool.pop()
        self.gl.glQueryCounter(q, GL_TIMESTAMP)
        self.used.append((name, q))

    def collect(self):
        """{segment name: ms} where segment k = mark k-1 -> mark k (the first mark is the start, its name unused)."""
        out = {}
        if not self.ok or not self.used:
            return out
        v = ctypes.c_uint64()
        ts = []
        for name, q in self.used:
            self.gl.glGetQueryObjectui64v(q, GL_QUERY_RESULT, ctypes.byref(v))
            ts.append((name, v.value))
            self.pool.append(q)
        self.used = []
        for (_, a), (name, b) in zip(ts, ts[1:]):
            out[name] = out.get(name, 0.0) + (b - a)*1e-6
        return out
