"""tvdsl.expr -- the TV language's safe arithmetic expression evaluator.

No Python eval anywhere: a tokenizer + Pratt parser builds a small AST which is compiled to closures.
Values are floats, points (2-tuples of floats) or lists (of values).  Only the names in FUNCS / constants and the
variables the caller binds are reachable.  Every failure evaluates to NaN (never raises at eval time).

    e = compile_expr("sin(2*pi*t/4) * r")          # -> Expr (raises ExprError on a syntax error)
    e.eval(env)                                   # env: Env (vars dict + optional scene hooks)
    e.const                                       # True when it depends on no variable (folded)
    e.names                                       # free variable names it reads
"""
import math

MAX_LEN = 600
MAX_DEPTH = 48
MAX_LIST = 400
NAN = float("nan")


class ExprError(ValueError):
    pass


class Env:
    """variables + hooks to the scene (object positions, path points) for pos()/pt()/end()"""
    __slots__ = ("vars", "scene")

    def __init__(self, vars=None, scene=None):
        self.vars = vars if vars is not None else {}
        self.scene = scene


# ---------------------------------------------------------------------------------------------------- value helpers
def _isnum(v):
    return isinstance(v, float) or isinstance(v, int)


def _f(v):
    """coerce to float (a point -> its x is NOT done: NaN)"""
    if isinstance(v, float):
        return v
    if isinstance(v, (int, bool)):
        return float(v)
    return NAN


def _pt(v):
    if isinstance(v, tuple) and len(v) == 2:
        return v
    return (NAN, NAN)


def _bin(op, a, b):
    ta, tb = isinstance(a, tuple), isinstance(b, tuple)
    if not ta and not tb:
        a, b = _f(a), _f(b)
        try:
            if op == "+": return a + b
            if op == "-": return a - b
            if op == "*": return a * b
            if op == "/": return a / b if b != 0 else (NAN if a == 0 else math.copysign(math.inf, a) * math.copysign(1, b))
            if op == "^":
                if abs(b) > 64 and abs(a) > 1.0001:
                    return NAN
                r = a ** b
                return r if isinstance(r, float) else NAN
            if op == "%": return a - b * math.floor(a / b) if b != 0 else NAN
        except (OverflowError, ValueError, ZeroDivisionError):
            return NAN
        return NAN
    if ta and tb and len(a) == 2 and len(b) == 2:
        if op == "+": return (a[0] + b[0], a[1] + b[1])
        if op == "-": return (a[0] - b[0], a[1] - b[1])
        if op == "*": return a[0] * b[0] + a[1] * b[1]       # dot product
        return (NAN, NAN)
    if ta and not tb:
        b = _f(b)
        if op == "*": return (a[0] * b, a[1] * b)
        if op == "/": return (a[0] / b, a[1] / b) if b else (NAN, NAN)
        return (NAN, NAN)
    if tb and not ta:
        a = _f(a)
        if op == "*": return (b[0] * a, b[1] * a)
        return (NAN, NAN)
    return NAN


def _cmp(op, a, b):
    a, b = _f(a), _f(b)
    if op == "<": return 1.0 if a < b else 0.0
    if op == ">": return 1.0 if a > b else 0.0
    if op == "<=": return 1.0 if a <= b else 0.0
    if op == ">=": return 1.0 if a >= b else 0.0
    if op == "==": return 1.0 if abs(a - b) < 1e-9 else 0.0
    if op == "!=": return 1.0 if abs(a - b) >= 1e-9 else 0.0
    return NAN


# ---------------------------------------------------------------------------------------------------- functions
def _smooth(x):
    x = 0.0 if x < 0 else 1.0 if x > 1 else x
    return x * x * (3 - 2 * x)


def _clamp(x, a=0.0, b=1.0):
    x, a, b = _f(x), _f(a), _f(b)
    return a if x < a else b if x > b else x


def _lerp(a, b, u):
    u = _f(u)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
    return _f(a) + (_f(b) - _f(a)) * u


def _mod(a, b):
    a, b = _f(a), _f(b)
    if b == 0 or b != b:
        return NAN
    return a - b * math.floor(a / b)


def _T(env):
    return _f(env.vars.get("t", 0.0))


def f_ramp(env, t0=0.0, dur=1.0, tt=None):
    """0 before t0, smooth rise to 1 over dur seconds"""
    tt = _T(env) if tt is None else _f(tt)
    d = max(1e-6, _f(dur))
    return _smooth((tt - _f(t0)) / d)


def f_lin(env, t0=0.0, dur=1.0, tt=None):
    """like ramp but linear (constant speed: tracing, moving at uniform speed)"""
    tt = _T(env) if tt is None else _f(tt)
    d = max(1e-6, _f(dur))
    return _clamp((tt - _f(t0)) / d)


def f_pulse(env, t0=0.0, dur=1.0, tt=None):
    """0 -> 1 -> 0 bump over [t0, t0+dur]"""
    tt = _T(env) if tt is None else _f(tt)
    d = max(1e-6, _f(dur))
    x = (tt - _f(t0)) / d
    return math.sin(math.pi * x) ** 2 if 0 <= x <= 1 else 0.0


def f_osc(env, period=2.0, tt=None):
    tt = _T(env) if tt is None else _f(tt)
    p = _f(period) or 1.0
    return math.sin(2 * math.pi * tt / p)


def f_saw(env, period=2.0, tt=None):
    tt = _T(env) if tt is None else _f(tt)
    p = _f(period) or 1.0
    return _mod(tt, p) / p


def f_tri(env, period=2.0, tt=None):
    s = f_saw(env, period, tt)
    return 1 - abs(2 * s - 1)


def f_steps(env, dur, *vals):
    """hold each value for dur seconds, easing to the next in the last 30% of the slot; clamps at the last value.
    Works on numbers and points: steps(1.5, (0,0), (2,0), (2,2))"""
    if not vals:
        return NAN
    d = max(1e-6, _f(dur))
    x = _T(env) / d
    if x <= 0:
        return vals[0]
    k = int(math.floor(x))
    if k >= len(vals) - 1:
        return vals[-1]
    fr = x - k
    u = _smooth((fr - 0.7) / 0.3)
    return _lerp(vals[k], vals[k + 1], u)


def f_cycle(env, dur, *vals):
    """like steps but loops back to the first value"""
    if not vals:
        return NAN
    d = max(1e-6, _f(dur))
    n = len(vals)
    x = _mod(_T(env) / d, n)
    if x != x:
        return vals[0]
    k = int(x)
    fr = x - k
    u = _smooth((fr - 0.7) / 0.3)
    return _lerp(vals[k % n], vals[(k + 1) % n], u)


def f_pick(i, *vals):
    if not vals:
        return NAN
    i = _f(i)
    if i != i:
        return NAN
    k = int(round(i))
    return vals[max(0, min(len(vals) - 1, k))]


def f_at(lst, i):
    if isinstance(lst, list) and lst:
        i = _f(i)
        if i != i:
            return NAN
        return lst[max(0, min(len(lst) - 1, int(round(i))))]
    return NAN


def f_len(lst):
    return float(len(lst)) if isinstance(lst, list) else NAN


def f_polar(r, deg):
    a = math.radians(_f(deg))
    r = _f(r)
    return (r * math.cos(a), r * math.sin(a))


def f_rot(p, deg, c=(0.0, 0.0)):
    p, c = _pt(p), _pt(c)
    a = math.radians(_f(deg))
    x, y = p[0] - c[0], p[1] - c[1]
    ca, sa = math.cos(a), math.sin(a)
    return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)


def f_dist(a, b):
    a, b = _pt(a), _pt(b)
    return math.hypot(a[0] - b[0], a[1] - b[1])


def f_dir(a, b=None):
    """unit vector: dir(deg) or dir(from, to)"""
    if b is None:
        r = math.radians(_f(a))
        return (math.cos(r), math.sin(r))
    a, b = _pt(a), _pt(b)
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n else (NAN, NAN)


def f_angle(a, b=None):
    """degrees: angle(p) of a vector, or angle(from, to)"""
    if b is None:
        p = _pt(a)
        return math.degrees(math.atan2(p[1], p[0]))
    a, b = _pt(a), _pt(b)
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def f_mid(a, b):
    return _lerp(_pt(a), _pt(b), 0.5)


def f_norm(p):
    p = _pt(p)
    return math.hypot(p[0], p[1])


def f_perp(p):
    p = _pt(p)
    return (-p[1], p[0])


# optics / physics helpers (generic formulas, not scenes)
def f_lensmaker(n, R1, R2, d=0.0):
    """focal length from the lensmaker's equation 1/f = (n-1)(1/R1 - 1/R2 + (n-1)d/(n R1 R2)); R = 0 means flat"""
    n, R1, R2, d = _f(n), _f(R1), _f(R2), _f(d)
    i1 = 0.0 if R1 == 0 else 1.0 / R1
    i2 = 0.0 if R2 == 0 else 1.0 / R2
    p = (n - 1) * (i1 - i2 + ((n - 1) * d * i1 * i2 / n if n else 0.0))
    return 1.0 / p if p else math.inf


def f_image(do, f):
    """thin-lens / mirror image distance from 1/do + 1/di = 1/f (do > 0 = real object); returns di"""
    do, f = _f(do), _f(f)
    q = 1.0 / f - 1.0 / do if do and f else NAN
    return 1.0 / q if q else math.inf


def f_snell(theta_deg, n1, n2):
    """refraction angle (degrees) from n1 sin a = n2 sin b; NaN on total internal reflection"""
    s = _f(n1) * math.sin(math.radians(_f(theta_deg))) / (_f(n2) or NAN)
    return math.degrees(math.asin(s)) if -1 <= s <= 1 else NAN


def f_critical(n1, n2):
    s = _f(n2) / (_f(n1) or NAN)
    return math.degrees(math.asin(s)) if 0 < s <= 1 else NAN


def f_cauchy(nm, n0=1.5, b=0.0042):
    """index of refraction at a wavelength in nm (Cauchy: n0 + b/um^2, normalised so 550 nm gives n0)"""
    um = _f(nm) / 1000.0
    return _f(n0) + _f(b) * (1 / (um * um) - 1 / 0.3025) if um else NAN


def _sin_theta(y, L):
    y, L = _f(y), _f(L)
    h = math.hypot(y, L)
    return y / h if h > 0 else NAN


def f_slit1(y, a, L, wl):
    """single slit of width a: (sin b / b)^2, b = pi a sin(theta) / wl"""
    s, a, wl = _sin_theta(y, L), _f(a), _f(wl)
    if not (wl > 0) or s != s:
        return NAN
    b = math.pi * a * s / wl
    return 1.0 if abs(b) < 1e-9 else (math.sin(b) / b) ** 2


def f_slit2(y, d, L, wl, a=0.0):
    """two slits d apart: cos^2(pi d sin(theta) / wl), times the single-slit envelope when the slit width a > 0"""
    s, d, wl = _sin_theta(y, L), _f(d), _f(wl)
    if not (wl > 0) or s != s:
        return NAN
    v = math.cos(math.pi * d * s / wl) ** 2
    return v * f_slit1(y, a, L, wl) if _f(a) > 0 else v


def f_grating(y, d, L, wl, N=5.0):
    """N slits d apart: (sin(N p) / (N sin p))^2, p = pi d sin(theta) / wl"""
    s, d, wl, N = _sin_theta(y, L), _f(d), _f(wl), int(max(1, min(1000, _f(N) if _f(N) == _f(N) else 1)))
    if not (wl > 0) or s != s:
        return NAN
    p = math.pi * d * s / wl
    den = N * math.sin(p)
    if abs(den) < 1e-9:
        return 1.0
    return (math.sin(N * p) / den) ** 2


def f_fringe(m, d, L, wl):
    """screen position of the m-th bright fringe of two slits / a grating: L tan(asin(m wl / d)) (NaN if none)"""
    m, d, L, wl = _f(m), _f(d), _f(L), _f(wl)
    if not d or abs(m * wl / d) > 1:
        return NAN
    return L * math.tan(math.asin(m * wl / d))


def f_fmt(x, nd=2):
    return x  # formatting is handled by text interpolation; in arithmetic it is the identity


# ---- 3D (orthographic projection with a view turn): molecules, orbitals, solids, vectors in space
_VSEPR = {
    1: [(1, 0, 0)],
    2: [(1, 0, 0), (-1, 0, 0)],
    3: [(0, 1, 0), (0.8660254, -0.5, 0), (-0.8660254, -0.5, 0)],
    4: [(0, 1, 0), (0.9428090, -0.3333333, 0), (-0.4714045, -0.3333333, 0.8164966), (-0.4714045, -0.3333333, -0.8164966)],
    5: [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-0.5, 0, 0.8660254), (-0.5, 0, -0.8660254)],
    6: [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)],
}


def _xyz(a, b, c):
    if isinstance(a, list) and len(a) == 3:
        s = _f(b) if _isnum(b) else 1.0
        return _f(a[0]) * s, _f(a[1]) * s, _f(a[2]) * s
    return _f(a), _f(b), _f(c)


def _yawpitch(env):
    v = env.vars if env is not None else {}
    yaw = _f(v.get("yaw", 30.0)) if _isnum(v.get("yaw", 30.0)) else 30.0
    pitch = _f(v.get("pitch", 15.0)) if _isnum(v.get("pitch", 15.0)) else 15.0
    return math.radians(yaw), math.radians(pitch)


def f_p3(env, a, b=NAN, c=NAN):
    """p3(x,y,z) or p3([x,y,z], scale): the 2D screen point of a 3D point, seen from the view angles `yaw` (turn
    about the vertical axis) and `pitch` (tilt) -- variables (param/let; defaults 30 and 15 degrees). z points at the viewer."""
    x, y, z = _xyz(a, b, c)
    ya, pa = _yawpitch(env)
    x1 = x * math.cos(ya) + z * math.sin(ya)
    z1 = -x * math.sin(ya) + z * math.cos(ya)
    return (x1, y * math.cos(pa) - z1 * math.sin(pa))


def f_z3(env, a, b=NAN, c=NAN):
    """depth of a 3D point toward the viewer (bigger = nearer): use for size/brightness/layer cues"""
    x, y, z = _xyz(a, b, c)
    ya, pa = _yawpitch(env)
    z1 = -x * math.sin(ya) + z * math.cos(ya)
    return y * math.sin(pa) + z1 * math.cos(pa)


def f_vsepr(n, i):
    """unit 3D direction [x,y,z] of electron domain i of n around a central atom (2 linear, 3 trigonal planar,
    4 tetrahedral, 5 trigonal bipyramidal, 6 octahedral); lone pairs are domains too (NH3 = 4 domains, 1 lone pair)"""
    d = _VSEPR.get(int(round(_f(n))))
    if d is None:
        return [NAN, NAN, NAN]
    v = d[int(round(_f(i))) % len(d)]
    return [float(v[0]), float(v[1]), float(v[2])]


def _need_scene(fn):
    def g(env, *a):
        sc = env.scene
        if sc is None:
            return (NAN, NAN)
        try:
            return fn(sc, *a)
        except Exception:  # noqa: BLE001
            return (NAN, NAN)
    return g


def _ref(a):
    return a if isinstance(a, str) else None


FUNCS = {
    # name: (callable, min_args, max_args, wants_env)
    "sin": (lambda x: math.sin(_f(x)), 1, 1, False),
    "cos": (lambda x: math.cos(_f(x)), 1, 1, False),
    "tan": (lambda x: math.tan(_f(x)), 1, 1, False),
    "asin": (lambda x: math.asin(_f(x)) if -1 <= _f(x) <= 1 else NAN, 1, 1, False),
    "acos": (lambda x: math.acos(_f(x)) if -1 <= _f(x) <= 1 else NAN, 1, 1, False),
    "atan": (lambda x: math.atan(_f(x)), 1, 1, False),
    "atan2": (lambda y, x: math.atan2(_f(y), _f(x)), 2, 2, False),
    "sind": (lambda x: math.sin(math.radians(_f(x))), 1, 1, False),
    "asind": (lambda x: math.degrees(math.asin(_f(x))) if -1 <= _f(x) <= 1 else NAN, 1, 1, False),
    "acosd": (lambda x: math.degrees(math.acos(_f(x))) if -1 <= _f(x) <= 1 else NAN, 1, 1, False),
    "atand": (lambda x: math.degrees(math.atan(_f(x))), 1, 1, False),
    "atan2d": (lambda y, x: math.degrees(math.atan2(_f(y), _f(x))), 2, 2, False),
    "cosd": (lambda x: math.cos(math.radians(_f(x))), 1, 1, False),
    "tand": (lambda x: math.tan(math.radians(_f(x))), 1, 1, False),
    "sinh": (lambda x: math.sinh(_f(x)) if abs(_f(x)) < 700 else NAN, 1, 1, False),
    "cosh": (lambda x: math.cosh(_f(x)) if abs(_f(x)) < 700 else NAN, 1, 1, False),
    "tanh": (lambda x: math.tanh(_f(x)), 1, 1, False),
    "exp": (lambda x: math.exp(_f(x)) if _f(x) < 700 else NAN, 1, 1, False),
    "log": (lambda x, b=None: (math.log(_f(x)) if b is None else math.log(_f(x), _f(b))) if _f(x) > 0 else NAN, 1, 2, False),
    "ln": (lambda x: math.log(_f(x)) if _f(x) > 0 else NAN, 1, 1, False),
    "log10": (lambda x: math.log10(_f(x)) if _f(x) > 0 else NAN, 1, 1, False),
    "sqrt": (lambda x: math.sqrt(_f(x)) if _f(x) >= 0 else NAN, 1, 1, False),
    "abs": (lambda x: abs(_f(x)) if not isinstance(x, tuple) else f_norm(x), 1, 1, False),
    "sign": (lambda x: math.copysign(1.0, _f(x)) if _f(x) else 0.0, 1, 1, False),
    "floor": (lambda x: float(math.floor(_f(x))) if math.isfinite(_f(x)) else NAN, 1, 1, False),
    "ceil": (lambda x: float(math.ceil(_f(x))) if math.isfinite(_f(x)) else NAN, 1, 1, False),
    "round": (lambda x, n=0: float(round(_f(x), int(_f(n)))) if math.isfinite(_f(x)) else NAN, 1, 2, False),
    "min": (lambda *a: min(_f(x) for x in a), 1, 32, False),
    "max": (lambda *a: max(_f(x) for x in a), 1, 32, False),
    "clamp": (_clamp, 1, 3, False),
    "lerp": (_lerp, 3, 3, False),
    "mix": (_lerp, 3, 3, False),
    "step": (lambda a, x: 1.0 if _f(x) >= _f(a) else 0.0, 2, 2, False),
    "smoothstep": (lambda a, b, x: _smooth((_f(x) - _f(a)) / ((_f(b) - _f(a)) or 1e-9)), 3, 3, False),
    "mod": (_mod, 2, 2, False),
    "hypot": (lambda a, b: math.hypot(_f(a), _f(b)), 2, 2, False),
    "deg": (lambda x: math.degrees(_f(x)), 1, 1, False),
    "rad": (lambda x: math.radians(_f(x)), 1, 1, False),
    "if": (lambda c, a, b: a if (_f(c) and _f(c) == _f(c)) else b, 3, 3, False),
    "pick": (f_pick, 2, 64, False),
    "at": (f_at, 2, 2, False),
    "len": (f_len, 1, 1, False),
    "isnan": (lambda x: 0.0 if _f(x) == _f(x) else 1.0, 1, 1, False),
    # time / animation (read t from the environment)
    "ramp": (f_ramp, 0, 3, True),
    "lin": (f_lin, 0, 3, True),
    "pulse": (f_pulse, 0, 3, True),
    "osc": (f_osc, 0, 2, True),
    "saw": (f_saw, 0, 2, True),
    "tri": (f_tri, 0, 2, True),
    "steps": (f_steps, 2, 64, True),
    "cycle": (f_cycle, 2, 64, True),
    # geometry
    "X": (lambda p: _pt(p)[0], 1, 1, False),
    "Y": (lambda p: _pt(p)[1], 1, 1, False),
    "P": (lambda x, y: (_f(x), _f(y)), 2, 2, False),
    "polar": (f_polar, 2, 2, False),
    "rot": (f_rot, 2, 3, False),
    "dist": (f_dist, 2, 2, False),
    "dir": (f_dir, 1, 2, False),
    "angle": (f_angle, 1, 2, False),
    "mid": (f_mid, 2, 2, False),
    "norm": (f_norm, 1, 1, False),
    "perp": (f_perp, 1, 1, False),
    # 3D
    "p3": (f_p3, 1, 3, True),
    "z3": (f_z3, 1, 3, True),
    "vsepr": (f_vsepr, 2, 2, False),
    # physics helpers
    "lensmaker": (f_lensmaker, 3, 4, False),
    "image": (f_image, 2, 2, False),
    "snell": (f_snell, 3, 3, False),
    "critical": (f_critical, 2, 2, False),
    "cauchy": (f_cauchy, 1, 3, False),
    "fmt": (f_fmt, 1, 2, False),
    # scene references (object ids are bare names resolved by the scene when not variables)
    "pos": (_need_scene(lambda sc, a: sc.ref_pos(_ref(a))), 1, 1, True),
    "pt": (_need_scene(lambda sc, a, u=0.5: sc.ref_point(_ref(a), _f(u))), 1, 2, True),
    "end": (_need_scene(lambda sc, a: sc.ref_point(_ref(a), 1.0)), 1, 1, True),
    "start": (_need_scene(lambda sc, a: sc.ref_point(_ref(a), 0.0)), 1, 1, True),
    "plen": (_need_scene(lambda sc, a: sc.ref_length(_ref(a))), 1, 1, True),
    "hit": (_need_scene(lambda sc, a, k=0: sc.ref_vertex(_ref(a), _f(k))), 1, 2, True),
    "node": (_need_scene(lambda sc, a, k=0: sc.ref_node(_ref(a), _f(k))), 2, 2, True),
    # diffraction / interference intensities (0..1, 1 at the centre). y = position on the screen,
    # L = slits-to-screen distance, d = slit spacing, a = slit width, wl = wavelength: all in ONE length unit.
    "slit1": (lambda y, a, L, wl: f_slit1(y, a, L, wl), 4, 4, False),
    "slit2": (lambda y, d, L, wl, a=0.0: f_slit2(y, d, L, wl, a), 4, 5, False),
    "grating": (lambda y, d, L, wl, N=5.0: f_grating(y, d, L, wl, N), 4, 5, False),
    "fringe": (lambda m, d, L, wl: f_fringe(m, d, L, wl), 4, 4, False),
}
CONSTS = {"pi": math.pi, "tau": 2 * math.pi, "e": math.e, "inf": math.inf, "true": 1.0, "false": 0.0, "π": math.pi}
# functions whose first argument may be an object id (a bare name that is not a variable)
REF_FUNCS = {"pos", "pt", "end", "start", "plen", "hit", "node"}


# ---------------------------------------------------------------------------------------------------- tokenizer
def _tokens(s):
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t":
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and s[i + 1].isdigit()):
            j = i
            while j < n and (s[j].isdigit() or s[j] == "."):
                j += 1
            if j < n and s[j] in "eE" and j + 1 < n and (s[j + 1].isdigit() or (s[j + 1] in "+-" and j + 2 < n and s[j + 2].isdigit())):
                j += 2
                while j < n and s[j].isdigit():
                    j += 1
            try:
                out.append(("num", float(s[i:j])))
            except ValueError:
                raise ExprError(f"bad number {s[i:j]!r}")
            i = j
            continue
        if c in "'\"‘“":                     # a short string literal: text choices in labels, c ? 'TIR' : 'refracts'
            close = {"'": "'", '"': '"', "‘": "’", "“": "”"}[c]
            j = s.find(close, i + 1)
            if j < 0 or j - i > 80:
                raise ExprError("bad string")
            out.append(("str", s[i + 1:j]))
            i = j + 1
            continue
        if c.isalpha() or c == "_" or c in "πθλφωαβγδμσΔΩ":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_" or s[j] in "πθλφωαβγδμσΔΩ'"):
                j += 1
            if s[i:j].startswith("__"):
                raise ExprError("bad name")
            out.append(("name", s[i:j]))
            i = j
            continue
        two = s[i:i + 2]
        if two in ("<=", ">=", "==", "!=", "**", "&&", "||"):
            out.append(("op", "^" if two == "**" else two))
            i += 2
            continue
        if c in "+-*/^%(),<>[]?:!×·−":
            out.append(("op", {"×": "*", "·": "*", "−": "-"}.get(c, c)))
            i += 1
            continue
        raise ExprError(f"unexpected {c!r}")
    out.append(("end", None))
    return out


# ---------------------------------------------------------------------------------------------------- parser -> closures
# node = (fn(env) -> value, const: bool, names: frozenset)
BP = {"||": 1, "&&": 2, "<": 4, ">": 4, "<=": 4, ">=": 4, "==": 4, "!=": 4, "+": 5, "-": 5, "*": 6, "/": 6, "%": 6, "^": 8}


class _P:
    def __init__(self, toks):
        self.t = toks
        self.i = 0
        self.depth = 0

    def peek(self):
        return self.t[self.i]

    def take(self):
        tk = self.t[self.i]
        self.i += 1
        return tk

    def expect(self, op):
        tk = self.take()
        if tk != ("op", op):
            raise ExprError(f"expected {op!r}")

    def expr(self, rbp=0):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise ExprError("too deeply nested")
        left = self.prefix()
        while True:
            tk = self.peek()
            if tk[0] == "op" and tk[1] == "?" and rbp < 1:
                self.take()
                a = self.expr(0)
                self.expect(":")
                b = self.expr(0)
                left = _cond(left, a, b)
                continue
            if tk[0] == "name" and tk[1] in ("mod", "and", "or"):        # infix words: i mod 6, a and b
                tk = ("op", {"mod": "%", "and": "&&", "or": "||"}[tk[1]])
                self.t[self.i] = tk
            if tk[0] != "op" or tk[1] not in BP:
                break
            op = tk[1]
            bp = BP[op]
            if bp <= rbp:
                break
            self.take()
            right = self.expr(bp - 1 if op == "^" else bp)
            left = _binnode(op, left, right)
        self.depth -= 1
        return left

    def prefix(self):
        tk = self.take()
        kind, v = tk
        if kind in ("num", "str"):
            return _constnode(v)
        if kind == "op" and v == "-":
            a = self.expr(7)
            return _unary(lambda x: _bin("*", x, -1.0), a)
        if kind == "op" and v == "+":
            return self.expr(7)
        if kind == "op" and v == "!":
            a = self.expr(7)
            return _unary(lambda x: 0.0 if _f(x) else 1.0, a)
        if kind == "op" and v == "(":
            items = [self.expr(0)]
            while self.peek() == ("op", ","):
                self.take()
                items.append(self.expr(0))
            self.expect(")")
            if len(items) == 1:
                return items[0]
            if len(items) == 2:
                return _tuplenode(items)
            raise ExprError("only 2-D points (x, y)")
        if kind == "op" and v == "[":
            items = []
            if self.peek() != ("op", "]"):
                items.append(self.expr(0))
                while self.peek() == ("op", ","):
                    self.take()
                    if self.peek() == ("op", "]"):
                        break
                    items.append(self.expr(0))
                    if len(items) > MAX_LIST:
                        raise ExprError("list too long")
            self.expect("]")
            return _listnode(items)
        if kind == "name":
            if self.peek() == ("op", "("):
                self.take()
                args = []
                if self.peek() != ("op", ")"):
                    args.append(self._arg(v, 0))
                    while self.peek() == ("op", ","):
                        self.take()
                        args.append(self._arg(v, len(args)))
                self.expect(")")
                return _callnode(v, args)
            if v in CONSTS:
                return _constnode(CONSTS[v])
            return _varnode(v)
        raise ExprError("unexpected end" if kind == "end" else f"unexpected {v!r}")

    def _arg(self, fname, k):
        if fname in REF_FUNCS and k == 0:
            tk = self.peek()
            nxt = self.t[self.i + 1] if self.i + 1 < len(self.t) else ("end", None)
            if tk[0] == "name" and nxt[0] == "op" and nxt[1] in (",", ")"):
                self.take()
                name = tk[1]
                return (lambda env, name=name: name, True, frozenset())
        return self.expr(0)


def _constnode(v):
    return (lambda env, v=v: v, True, frozenset())


def _fold(node):
    fn, c, names = node
    if c:
        try:
            v = fn(Env())
        except Exception:  # noqa: BLE001
            v = NAN
        return (lambda env, v=v: v, True, frozenset())
    return node


def _varnode(name):
    def fn(env, name=name):
        v = env.vars.get(name)
        if v is None:
            return NAN
        return v
    return (fn, False, frozenset([name]))


def _unary(op, a):
    fa = a[0]
    return _fold((lambda env: op(fa(env)), a[1], a[2]))


def _binnode(op, a, b):
    fa, fb = a[0], b[0]
    if op in ("<", ">", "<=", ">=", "==", "!="):
        fn = lambda env: _cmp(op, fa(env), fb(env))
    elif op == "&&":
        fn = lambda env: 1.0 if (_f(fa(env)) and _f(fb(env))) else 0.0
    elif op == "||":
        fn = lambda env: 1.0 if (_f(fa(env)) or _f(fb(env))) else 0.0
    else:
        fn = lambda env: _bin(op, fa(env), fb(env))
    return _fold((fn, a[1] and b[1], a[2] | b[2]))


def _cond(c, a, b):
    fc, fa, fb = c[0], a[0], b[0]

    def fn(env):
        x = _f(fc(env))
        return fa(env) if (x and x == x) else fb(env)
    return _fold((fn, c[1] and a[1] and b[1], c[2] | a[2] | b[2]))


def _tuplenode(items):
    fx, fy = items[0][0], items[1][0]
    return _fold((lambda env: (_f(fx(env)), _f(fy(env))), items[0][1] and items[1][1], items[0][2] | items[1][2]))


def _listnode(items):
    fs = [it[0] for it in items]
    c = all(it[1] for it in items)
    names = frozenset().union(*[it[2] for it in items]) if items else frozenset()
    return _fold((lambda env: [f(env) for f in fs], c, names))


def _rootnode(args):
    """root(EXPR_OF_x, a, b): the x in [a, b] where EXPR is 0 (first sign change, bisection) -- intersections,
    equilibria, break-even points stay exact when params change. root(EXPR, a, b, k) = the k-th root (0-based)."""
    fe, fa, fb = args[0][0], args[1][0], args[2][0]
    fk = args[3][0] if len(args) > 3 else None
    names = frozenset().union(*[a[2] for a in args]) - {"x"}

    def fn(env):
        a, b = _f(fa(env)), _f(fb(env))
        k = int(_f(fk(env))) if fk is not None else 0
        if not (math.isfinite(a) and math.isfinite(b)) or a == b:
            return NAN
        v = dict(env.vars)
        e2 = Env(v, env.scene)

        def g(x):
            v["x"] = x
            return _f(fe(e2))
        n = 96
        xs = [a + (b - a) * i / n for i in range(n + 1)]
        ys = [g(x) for x in xs]
        found = -1
        for i in range(n):
            y0, y1 = ys[i], ys[i + 1]
            if not (y0 == y0 and y1 == y1):
                continue
            if y0 == 0 or y0 * y1 < 0 or (i == n - 1 and y1 == 0):
                found += 1
                if found < k:
                    continue
                if y0 == 0:
                    return xs[i]
                if y1 == 0:
                    return xs[i + 1]
                lo, hi, flo = xs[i], xs[i + 1], y0
                for _ in range(48):
                    mid = (lo + hi) / 2
                    fm = g(mid)
                    if fm != fm:
                        break
                    if (fm < 0) == (flo < 0):
                        lo, flo = mid, fm
                    else:
                        hi = mid
                return (lo + hi) / 2
        return NAN
    return (fn, False, names)


def _callnode(name, args):
    if name == "root":
        if not (3 <= len(args) <= 4):
            raise ExprError("root() takes 3..4 arguments")
        return _rootnode(args)
    spec = FUNCS.get(name)
    if spec is None:
        # a user function (func NAME(args) = ...) resolved by the scene at eval time
        if len(args) > 8:
            raise ExprError(f"{name}() takes at most 8 arguments")
        fs = [a[0] for a in args]
        names = frozenset().union(*[a[2] for a in args]) if args else frozenset()

        def ucall(env, name=name):
            sc = env.scene
            if sc is None or not hasattr(sc, "call_user"):
                return NAN
            try:
                return sc.call_user(name, [f(env) for f in fs], env)
            except RecursionError:
                return NAN
        return (ucall, False, names | {"@user"})
    fn, lo, hi, wants_env = spec
    if not (lo <= len(args) <= hi):
        raise ExprError(f"{name}() takes {lo}..{hi} arguments")
    fs = [a[0] for a in args]
    names = frozenset().union(*[a[2] for a in args]) if args else frozenset()
    def user_over(env):
        # a program's own `func NAME(..)` wins over a built-in of the same name (fu5: `func tri(u)` was
        # silently the built-in triangle WAVE of t, so 60 gas particles sat in one corner)
        sc = env.scene
        fd = getattr(getattr(sc, "doc", None), "funcs", None) if sc is not None else None
        return sc if fd and name in fd else None

    if wants_env:
        def call(env):
            sc = user_over(env)
            try:
                if sc is not None:
                    return sc.call_user(name, [f(env) for f in fs], env)
                v = fn(env, *[f(env) for f in fs])
            except Exception:  # noqa: BLE001
                return NAN
            return v
        if name in ("ramp", "lin", "pulse", "osc", "saw", "tri", "steps", "cycle"):
            names = names | {"t"}
        return (call, False, names | {"@scene"} if name in REF_FUNCS else names)

    def call2(env):
        try:
            sc = user_over(env)
            if sc is not None:
                return sc.call_user(name, [f(env) for f in fs], env)
            return fn(*[f(env) for f in fs])
        except Exception:  # noqa: BLE001
            return NAN
    return _fold((call2, all(a[1] for a in args), names))


class Expr:
    __slots__ = ("src", "fn", "const", "names", "_v")

    def __init__(self, src, node):
        self.src = src
        self.fn, self.const, self.names = node
        self._v = self.fn(Env()) if self.const else None

    def eval(self, env):
        if self.const:
            return self._v
        try:
            return self.fn(env)
        except RecursionError:
            return NAN
        except Exception:  # noqa: BLE001
            return NAN

    def __repr__(self):
        return f"Expr({self.src!r})"


def compile_expr(src):
    if not isinstance(src, str):
        raise ExprError("not text")
    s = src.strip()
    if not s:
        raise ExprError("empty")
    if len(s) > MAX_LEN:
        raise ExprError("too long")
    p = _P(_tokens(s))
    node = p.expr(0)
    if p.peek()[0] != "end":
        raise ExprError(f"unexpected {p.peek()[1]!r}")
    return Expr(s, node)


def evaluate(src, **vars):
    """convenience: compile + eval with plain variables (tests)"""
    return compile_expr(src).eval(Env(dict(vars)))
