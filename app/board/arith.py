"""boardbrain: cheap arithmetic self-check for what the board brain writes.

Code can verify plain numeric arithmetic, so it does: a chain like `60+80=140`, `\\frac{36}{6} = 6`,
`730/140 \\approx 5.21` or a column sum/difference/product is evaluated exactly; a WRONG final number (a plain literal
at the end of a chain whose earlier parts agree) is replaced by the right one, anything else wrong is reported as a
warning. Expressions with letters/units are skipped (nothing to verify). General: no subject or phrase special cases.

    check_item(kind, content) -> (content, [notes])
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_REL = re.compile(r"\\approx|≈|=")
_NUM = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")


def _frac_sub(s):
    # \frac{a}{b} -> ((a)/(b)), innermost first
    pat = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(8):
        s2 = pat.sub(r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    return s


def tex_to_expr(seg: str):
    """LaTeX-ish numeric expression -> python expression over Fractions, or None when it is not pure arithmetic."""
    s = seg.strip()
    if not s or len(s) > 160:
        return None
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    s = s.replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("·", "*").replace("⋅", "*")
    s = s.replace("\\div", "/").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)          # thousands separators
    s = _frac_sub(s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^(\d)", r"**\1", s)
    s = s.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    s = re.sub(r"\s+", "", s)
    if not s or re.search(r"[^0-9.+\-*/()]", s):
        return None
    if not re.search(r"\d", s):
        return None
    s = re.sub(r"(?<=[\d)])\(", "*(", s)                     # implicit multiplication 60(5.21)
    s = re.sub(r"\)(?=[\d(])", ")*", s)
    if "**" in s and re.search(r"\*\*\(?\d{3,}", s):
        return None                                           # huge powers: not worth it
    s = re.sub(r"(\d+(?:\.\d+)?)", r"F('\1')", s)
    return s


def evaluate(seg: str):
    e = tex_to_expr(seg)
    if e is None:
        return None
    try:
        v = eval(e, {"__builtins__": {}}, {"F": Fraction})  # noqa: S307 — digits/operators/F only (checked above)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, int):
        v = Fraction(v)
    return v if isinstance(v, Fraction) else None


def _decimals(lit: str) -> int:
    lit = lit.strip()
    return len(lit.split(".")[1]) if "." in lit else 0


def _fmt(v: Fraction, like: str) -> str:
    d = _decimals(like)
    if v.denominator == 1 and d == 0:
        return str(v.numerator)
    q = Decimal(v.numerator) / Decimal(v.denominator)
    return str(q.quantize(Decimal(1).scaleb(-max(d, 2 if v.denominator != 1 else 0))))


def _close(a: Fraction, b: Fraction, rel: str, rhs_lit: str | None) -> bool:
    if a == b:
        return True
    tol = Fraction(0)
    if rhs_lit is not None and _NUM.match(rhs_lit):
        d = _decimals(rhs_lit)
        if d or rel != "=":
            tol = Fraction(1, 2 * 10 ** d) * Fraction(1001, 1000)
    if rel != "=":
        tol = max(tol, abs(a) / 100)
    return abs(a - b) <= tol


def check_chain(src: str):
    """-> (fixed_src, notes). Checks consecutive numeric parts of 'a = b ≈ c'; fixes a wrong trailing literal."""
    if "\\begin" in src or "&" in src or "\\\\" in src:
        return src, []
    parts, rels, last = [], [], 0
    depth = 0
    for m in _REL.finditer(src):
        pre = src[last:m.start()]
        depth += pre.count("{") - pre.count("}")
        if depth != 0:
            continue
        parts.append((last, m.start()))
        rels.append("=" if m.group(0) == "=" else "~")
        last = m.end()
    parts.append((last, len(src)))
    if len(parts) < 2:
        return src, []
    vals = [evaluate(src[a:b]) for a, b in parts]
    notes = []
    fixed = src
    for i in range(1, len(parts)):
        a, b = vals[i - 1], vals[i]
        if a is None or b is None:
            continue
        lit = src[parts[i][0]:parts[i][1]]
        if _close(a, b, rels[i - 1], lit):
            continue
        if i == len(parts) - 1 and _NUM.match(lit):
            new = _fmt(a, lit.strip())
            lead = lit[:len(lit) - len(lit.lstrip())]
            trail = lit[len(lit.rstrip()):]
            fixed = src[:parts[i][0]] + lead + new + trail
            notes.append("arith fixed: %s -> %s" % (lit.strip(), new))
        else:
            notes.append("arith mismatch: %s vs %s" % (src[parts[i - 1][0]:parts[i - 1][1]].strip(), lit.strip()))
    return fixed, notes


_PLAIN_EQ = re.compile(r"(?<![\w.$\\])(\(?-?\d[\d.,]*(?:\s*[-+*/×÷x·]\s*\(?\d[\d.,]*\)?)+\s*(?:=|≈)\s*-?\d[\d.,]*)(?![\w.]*\d)")


def check_text(src: str):
    """Inline $math$ segments and plain 'a + b = c' runs in a text item."""
    notes = []

    def fix_math(m):
        f, n = check_chain(m.group(1))
        notes.extend(n)
        return "$" + f + "$"
    out = re.sub(r"\$([^$]+)\$", fix_math, src)

    def fix_plain(m):
        seg = m.group(1).replace("x", "×") if re.search(r"\d\s*x\s*\d", m.group(1)) else m.group(1)
        f, n = check_chain(seg)
        notes.extend(n)
        return f if n and any(x.startswith("arith fixed") for x in n) else m.group(1)
    parts = re.split(r"(\$[^$]+\$)", out)
    out = "".join(p if p.startswith("$") else _PLAIN_EQ.sub(fix_plain, p) for p in parts)
    return out, notes


def check_column(src: str):
    """'347 ; x 28 ; --- ; 2776 ; 6940 ; --- ; 9716' -> fix the final row if it is the wrong sum/difference/product."""
    rows = [r.strip() for r in re.split(r"[;\n]", src)]
    idx = [i for i, r in enumerate(rows) if r]
    body = []
    for i in idx:
        r = rows[i]
        if re.fullmatch(r"[-_=—–]+", r):
            body.append(("rule", i, None, None))
        elif r.lower().startswith("c:"):
            continue
        else:
            m = re.match(r"^([+\-x×÷*−])\s*(.*)$", r)
            op = m.group(1) if m else ""
            num = (m.group(2) if m else r).replace(" ", "").replace(",", "")
            try:
                v = Decimal(num)
            except InvalidOperation:
                return src, []
            body.append(("row", i, {"x": "*", "×": "*", "*": "*", "−": "-"}.get(op, op), v))
    rules = [k for k, b in enumerate(body) if b[0] == "rule"]
    if not rules or body[-1][0] != "row":
        return src, []
    first_rule = rules[0]
    operands = [b for b in body[:first_rule] if b[0] == "row"]
    final = body[-1]
    if len(operands) < 2:
        return src, []
    ops = {b[2] for b in operands[1:]}
    if len(ops) != 1:
        return src, []
    op = ops.pop() or "+"
    vals = [b[3] for b in operands]
    if op == "+":
        want = sum(vals)
    elif op == "-":
        want = vals[0] - sum(vals[1:])
    elif op == "*":
        want = vals[0]
        for v in vals[1:]:
            want *= v
    else:
        return src, []
    if final[3] == want:
        return src, []
    old = rows[final[1]]
    new = format(want, "f")
    if "," in old:
        new = "{:,}".format(want)
    rows[final[1]] = new
    return " ; ".join(r for r in rows if r), ["arith fixed: column %s -> %s" % (old, new)]


def check_item(kind: str, content: str):
    try:
        if kind == "math":
            return check_chain(content)
        if kind == "text":
            return check_text(content)
        if kind == "column":
            return check_column(content)
    except Exception:  # noqa: BLE001 — a checker bug never blocks the board
        pass
    return content, []
