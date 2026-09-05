"""FLAME GLPS reader (PLAN §6 task 3.4).

FLAME (FRIB Linear Accelerator Modelling Engine) reads *GLPS* decks: a
``name: type, key = value, …;`` dialect close to MAD8's, with vectors, strings
and ``name: LINE = (…);`` / ``USE: name;``.  The grammar implemented here is a
faithful hand-port of FLAME's own ``src/glps.y`` + ``src/glps.l``:

======================  =======================================================
token                   lexer rule (``src/glps.l``)
======================  =======================================================
identifier              ``[A-Za-z]([A-Za-z0-9_:]*[A-Za-z0-9_])?`` — note the
                        embedded ``:`` (EPICS PV names), so ``a:b`` is ONE token
                        and an element definition needs the space in ``a: b``
number                  ``[0-9]+(\\.[0-9]*)?([eE][+-]?[0-9]+)?`` — no leading
                        ``.``, no sign (``-`` is a unary operator)
string                  ``"…"``, no escapes, no embedded newline
comment                 ``#`` to end of line
punctuation             ``= : ; ( ) [ ] , + * / -``   (**no** ``^``/``**``)
======================  =======================================================

Statements (``src/glps.y``)::

    assignment : KEYWORD '=' expr ';'
    element    : KEYWORD ':' KEYWORD properties ';'
    line       : KEYWORD ':' LINE '=' '(' line_list ')' ';'
    func       : KEYWORD '(' expr ')' ';'          # only print(…)
    command    : KEYWORD ';'                       # only END

``USE: cell;`` is *syntactically* an element named ``USE`` whose "type" is the
line name (FLAME finds it at ``src/config.cpp:283``).  Line lists take
expressions, so ``2*cell`` repeats and ``-cell`` reverses
(``src/glps_ops.cpp:236-238``); nested lines are spliced by FLAME but kept as IR
:class:`~lattix.ir.lattice.Line` references here, which ``flatten()`` reproduces.
Built-in functions: ``sin cos tan asin acos atan arcsin arccos arctan deg2rad
rad2deg file dir parse h5file`` (``src/glps_ops.cpp:216-244``).

Element table — verified against ``src/moment.cpp`` and ``sphinx_doc/element.rst``:

===============  ==========================================  =========================
FLAME type       parameters (units)                          IR
===============  ==========================================  =========================
``source``       ``vector_variable``/``matrix_variable``      Marker (start of line)
``marker``       —                                            Marker
``bpm``          —                                            Instrument(BPM)
``drift``        ``L`` [m]                                    Drift
``orbtrim``      ``theta_x``/``theta_y`` [rad] (``realpara``  Kicker
                 = 1: ``tm_xkick``/``tm_ykick`` [T·m]),
                 ``xyrotate`` [deg]
``quadrupole``   ``L`` [m], ``B2`` [T/m]                      Quadrupole ``Bn[1]``
``sextupole``    ``L`` [m], ``B3`` [T/m²]                     Sextupole ``Bn[2]``
``solenoid``     ``L`` [m], ``B`` [T]                         Solenoid ``Bsol``
``sbend``        ``L`` [m], ``phi``/``phi1``/``phi2``/        Bend (+ ``Bn[1]``)
                 ``dphi1``/``dphi2`` [deg], ``K`` [1/m²],
                 ``bg`` [1], ``ver`` (1 = vertical)
``rfcavity``     ``L`` [m], ``cavtype``, ``f`` [Hz],          RFCavity
                 ``phi`` [deg], ``scl_fac``, ``syncflag``
``stripper``     ``IonChargeStates``, ``NCharge``, …          Foil
``tmatrix``      ``matrix`` (7×7 flattened)                   Taylor
``edipole``      electrostatic bend                           Marker + DROPPED
``equad``        ``V`` [V], ``radius`` [m]                    Marker + DROPPED
===============  ==========================================  =========================

``K`` is a *normalized* gradient exactly like MAD-X's ``k1``: ``moment.cpp:1020``
reads it as ``K/sqr(MtoMM)`` (MtoMM = 1e3, i.e. 1/m² → 1/mm²) and
``moment_sup.cpp:228-230`` builds ``Kx = K + 1/ρ²``, ``Ky = −K``.  Misalignments
``dx dy`` [m] and ``pitch yaw roll`` [rad] map to ``BodyShiftP``; ``aper`` [m] is
a deck convention FLAME itself never reads (it appears in no ``conf().get`` call)
and is kept as a circular :class:`~lattix.ir.elements.ApertureP`.

Reference particle: FLAME is **per nucleon** — ``IonEs`` and ``IonEk`` are eV/u and
``IonChargeStates`` are charge-to-mass ratios Q/A (``Brho = β·IonW/(c·IonZ)``,
``src/flame/moment.h:57``).  The IR wants a total mass and an integer charge, so
Q and A are recovered as the smallest rational with denominator ≤ 400 that matches
Q/A, and ``Species(mass_eV = IonEs·A, charge = Q)``; a lattice with several charge
states keeps the first and records EQUIVALENT ``MULTI_CHARGE_STATE_FIRST``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    ApertureP,
    Bend,
    BodyShiftP,
    Drift,
    Element,
    Foil,
    Instrument,
    Kicker,
    Marker,
    Provenance,
    Quadrupole,
    RFCavity,
    Sextupole,
    Solenoid,
    Taylor,
)
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.normalize import gradient_from_k1, kick_from_bl
from lattix.ir.reference import ReferenceParticle, Species

#: FLAME's ``SampleFreqDefault`` (``src/flame/moment.h:17``) — the RF clock the
#: longitudinal coordinate φ [rad] is measured against, *not* a cavity frequency.
SAMPLE_FREQ_DEFAULT = 80.5e6

#: default nucleon mass used by the FRIB decks (``AMU`` in ``ALL_lattice.lat``).
AMU_eV = 931.49432e6

#: largest mass number a charge-to-mass ratio is resolved against.
MAX_MASS_NUMBER = 400

#: ``sim_type`` values FLAME implements (``src/moment.cpp:1526``, ``src/linear.cpp:374``).
SIM_TYPES = ("MomentMatrix", "TransferMatrix", "Vector")

#: keys that live on the *lattice* rather than on an element.
GLOBAL_KEYS = frozenset({
    "sim_type", "MpoleLevel", "EmitGrowth", "HdipoleFitMode", "IonEs", "IonEk", "IonW",
    "IonZ", "IonChargeStates", "NCharge", "SampleFreq", "Eng_Data_Dir", "AMU",
    "Stripper_IonChargeStates", "Stripper_NCharge", "cstate",
})

#: FLAME element types this reader knows (``src/moment.cpp:1526-1552``).
ELEMENT_TYPES = frozenset({
    "source", "marker", "bpm", "drift", "orbtrim", "sbend", "quadrupole", "sextupole",
    "solenoid", "rfcavity", "stripper", "edipole", "equad", "tmatrix", "generic",
})

#: misalignment keys (``sphinx_doc/element.rst``: dx/dy [m], pitch/yaw/roll [rad]).
MISALIGN_KEYS = ("dx", "dy", "pitch", "yaw", "roll")

_FUNCTIONS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan,
    "deg2rad": math.radians, "rad2deg": math.degrees,
}
#: functions whose argument is a path/string and which pass it through.
_PATH_FUNCTIONS = frozenset({"file", "dir", "parse", "h5file"})

#: the writer's provenance comment (PLAN §4.4, invariant I-15).
TAG_RE = re.compile(r'^\s*#\s*lattix:\s*name="([^"]*)"(?:\s+type="([^"]*)")?\s*$')


def parse_tags(text: str) -> dict[int, tuple[str, str | None]]:
    """``# lattix: name="…" type="…"`` comments, keyed by the line they annotate.

    The writer puts the tag on the line above the definition it belongs to, so a
    FLAME → IR → FLAME → IR round trip recovers the original name and type
    (invariant I-15) even though FLAME itself cannot spell either.
    """
    out: dict[int, tuple[str, str | None]] = {}
    for i, line in enumerate(text.splitlines(), start=1):
        m = TAG_RE.match(line)
        if m:
            out[i + 1] = (m.group(1), m.group(2))
    return out


# ---------------------------------------------------------------------------
# lexer
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"""
      (?P<comment>\#[^\n]*)
    | (?P<space>[ \t\r\n]+)
    | (?P<name>[A-Za-z](?:[A-Za-z0-9_:]*[A-Za-z0-9_])?)
    | (?P<num>[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)
    | (?P<string>"[^"\n\r]*")
    | (?P<punct>[=:;()\[\],+*/-])
    """,
    re.VERBOSE,
)


class GLPSError(ValueError):
    """A GLPS deck FLAME itself would reject."""

    def __init__(self, message: str, line: int | None = None):
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line


@dataclass(frozen=True)
class Token:
    kind: str      # name | num | string | punct | eof
    text: str
    line: int

    def is_punct(self, ch: str) -> bool:
        return self.kind == "punct" and self.text == ch


def tokenize(text: str) -> list[Token]:
    """GLPS tokens, comments and whitespace removed (``src/glps.l``)."""
    out: list[Token] = []
    pos, line = 0, 1
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            ch = text[pos]
            raise GLPSError(f"invalid character {ch!r} ({ord(ch)})", line)
        kind = m.lastgroup
        raw = m.group()
        if kind in ("space", "comment"):
            line += raw.count("\n")
        else:
            out.append(Token(kind, raw[1:-1] if kind == "string" else raw, line))
        pos = m.end()
    out.append(Token("eof", "", line))
    return out


# ---------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------
@dataclass
class Node:
    """A GLPS expression: ``op`` is ``num``/``str``/``var``/``vec`` or an operator."""

    op: str
    value: float | str | None = None
    args: list[Node] = field(default_factory=list)


@dataclass
class LineRef:
    """One entry of a ``LINE = (…)`` list after ``n*x`` / ``-x`` reduction."""

    name: str
    repeat: int = 1
    reverse: bool = False


class _Parser:
    """Recursive-descent parser mirroring ``src/glps.y``."""

    def __init__(self, tokens: list[Token]):
        self.t = tokens
        self.i = 0

    # -- token helpers ---------------------------------------------------
    @property
    def cur(self) -> Token:
        return self.t[self.i]

    def take(self) -> Token:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, ch: str) -> Token:
        tok = self.cur
        if not tok.is_punct(ch):
            raise GLPSError(f"expected {ch!r}, got {tok.text or 'end of file'!r}", tok.line)
        return self.take()

    # -- expressions -----------------------------------------------------
    def expr(self) -> Node:
        return self._additive()

    def _additive(self) -> Node:
        node = self._multiplicative()
        while self.cur.kind == "punct" and self.cur.text in "+-":
            op = self.take().text
            node = Node(op, args=[node, self._multiplicative()])
        return node

    def _multiplicative(self) -> Node:
        node = self._unary()
        while self.cur.kind == "punct" and self.cur.text in "*/":
            op = self.take().text
            node = Node(op, args=[node, self._unary()])
        return node

    def _unary(self) -> Node:
        if self.cur.is_punct("-"):
            self.take()
            return Node("neg", args=[self._unary()])
        return self._atom()

    def _atom(self) -> Node:
        tok = self.take()
        if tok.kind == "num":
            return Node("num", float(tok.text))
        if tok.kind == "string":
            return Node("str", tok.text)
        if tok.kind == "name":
            if self.cur.is_punct("("):
                self.take()
                arg = self.expr()
                self.expect(")")
                return Node(tok.text, args=[arg])
            return Node("var", tok.text)
        if tok.is_punct("("):
            node = self.expr()
            self.expect(")")
            return node
        if tok.is_punct("["):
            items: list[Node] = []
            if not self.cur.is_punct("]"):
                items.append(self.expr())
                while self.cur.is_punct(","):
                    self.take()
                    if self.cur.is_punct("]"):     # trailing comma
                        break
                    items.append(self.expr())
            self.expect("]")
            return Node("vec", args=items)
        raise GLPSError(f"unexpected {tok.text or 'end of file'!r}", tok.line)


@dataclass
class Statement:
    kind: str                      # assign | element | line | use | command | func
    name: str
    type: str = ""
    line: int = 0
    value: Node | None = None
    props: list[tuple[str, Node]] = field(default_factory=list)
    items: list[Node] = field(default_factory=list)


def parse_statements(text: str) -> list[Statement]:
    """Split a GLPS deck into statements without evaluating anything."""
    p = _Parser(tokenize(text))
    out: list[Statement] = []
    while p.cur.kind != "eof":
        tok = p.take()
        if tok.kind != "name":
            raise GLPSError(f"statement must start with a name, got {tok.text!r}", tok.line)
        if p.cur.is_punct("="):                                   # assignment
            p.take()
            value = p.expr()
            p.expect(";")
            out.append(Statement("assign", tok.text, line=tok.line, value=value))
        elif p.cur.is_punct(":"):
            p.take()
            type_tok = p.take()
            if type_tok.kind != "name":
                raise GLPSError(f"expected a type name, got {type_tok.text!r}", type_tok.line)
            if p.cur.is_punct("="):                               # LINE definition
                if type_tok.text.upper() != "LINE":               # glps_parser.cpp:459
                    raise GLPSError(f"line-like definition {tok.text!r} with "
                                    f"{type_tok.text!r} instead of 'LINE'", tok.line)
                p.take()
                p.expect("(")
                items: list[Node] = []
                if not p.cur.is_punct(")"):
                    items.append(p.expr())
                    while p.cur.is_punct(","):
                        p.take()
                        if p.cur.is_punct(")"):                   # trailing comma
                            break
                        items.append(p.expr())
                p.expect(")")
                p.expect(";")
                out.append(Statement("line", tok.text, line=tok.line, items=items))
            else:                                                 # element (or USE)
                props: list[tuple[str, Node]] = []
                while p.cur.is_punct(","):
                    p.take()
                    key = p.take()
                    if key.kind != "name":
                        raise GLPSError(f"expected a parameter name, got {key.text!r}", key.line)
                    p.expect("=")
                    props.append((key.text, p.expr()))
                p.expect(";")
                kind = "use" if tok.text == "USE" and not props else "element"
                out.append(Statement(kind, tok.text, type_tok.text, tok.line, props=props))
        elif p.cur.is_punct("("):                                 # print(…);
            p.take()
            value = p.expr()
            p.expect(")")
            p.expect(";")
            out.append(Statement("func", tok.text, line=tok.line, value=value))
        elif p.cur.is_punct(";"):                                 # END;
            p.take()
            out.append(Statement("command", tok.text, line=tok.line))
        else:
            raise GLPSError(f"unexpected {p.cur.text or 'end of file'!r} after {tok.text!r}",
                            p.cur.line)
    return out


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(node: Node, variables: dict[str, object], *, line: int | None = None) -> object:
    """Evaluate a GLPS expression to a float, str or list."""
    op = node.op
    if op == "num":
        return node.value
    if op == "str":
        return node.value
    if op == "vec":
        return [evaluate(a, variables, line=line) for a in node.args]
    if op == "var":
        name = str(node.value)
        if name not in variables:
            raise GLPSError(f"undefined variable {name!r}", line)
        return variables[name]
    if op == "neg":
        return -_number(evaluate(node.args[0], variables, line=line), line)
    if op in ("+", "-", "*", "/"):
        a = _number(evaluate(node.args[0], variables, line=line), line)
        b = _number(evaluate(node.args[1], variables, line=line), line)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if b == 0.0:
            raise GLPSError("division by zero", line)
        return a / b
    if op in _FUNCTIONS:
        return _FUNCTIONS[op](_number(evaluate(node.args[0], variables, line=line), line))
    if op in _PATH_FUNCTIONS:
        return evaluate(node.args[0], variables, line=line)
    raise GLPSError(f"undefined function {op!r}", line)


def _number(value: object, line: int | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GLPSError(f"expected a number, got {value!r}", line)
    return float(value)


def expr_text(node: Node) -> str:
    """Re-render an expression (used to keep ``33.0/238.0`` legible in ``meta``)."""
    op = node.op
    if op == "num":
        return _fmt(float(node.value))
    if op == "str":
        return f'"{node.value}"'
    if op == "var":
        return str(node.value)
    if op == "vec":
        return "[" + ", ".join(expr_text(a) for a in node.args) + "]"
    if op == "neg":
        return f"-{expr_text(node.args[0])}"
    if op in ("+", "-", "*", "/"):
        return f"({expr_text(node.args[0])} {op} {expr_text(node.args[1])})"
    return f"{op}({', '.join(expr_text(a) for a in node.args)})"


def _fmt(x: float) -> str:
    s = f"{x:.15g}"
    return "0" if s in ("-0", "-0.0") else s


def _line_refs(items: list[Node], line: int) -> list[LineRef]:
    """Reduce a LINE list to (name, repeat, reverse) triples."""
    out: list[LineRef] = []
    for node in items:
        repeat, reverse = 1, False
        cur = node
        while True:
            if cur.op == "neg":
                reverse = not reverse
                cur = cur.args[0]
            elif cur.op == "*":
                a, b = cur.args
                if a.op == "num":
                    repeat *= int(round(float(a.value)))
                    cur = b
                elif b.op == "num":
                    repeat *= int(round(float(b.value)))
                    cur = a
                else:
                    raise GLPSError("a line product needs a numeric factor", line)
            else:
                break
        if cur.op != "var":
            raise GLPSError(f"lines cannot be built from {cur.op!r}", line)
        out.append(LineRef(str(cur.value), repeat, reverse))
    return out


def charge_and_mass_number(q_over_a: float, max_a: int = MAX_MASS_NUMBER) -> tuple[int, int, float]:
    """Smallest (Q, A) with ``Q/A ≈ q_over_a``; returns (Q, A, relative error).

    FLAME never stores A, only the charge-to-mass ratio, so the IR's integer
    charge and total mass are recovered from the rational form the decks use
    (``IonZ = 33.0/238.0`` → Q = 33, A = 238).
    """
    if q_over_a == 0.0:
        return 0, 1, 0.0
    frac = Fraction(abs(q_over_a)).limit_denominator(max_a)
    q, a = frac.numerator, frac.denominator
    if q == 0:
        return 0, 1, 1.0
    err = abs(q / a - abs(q_over_a)) / abs(q_over_a)
    return (q if q_over_a > 0 else -q), a, err


# ---------------------------------------------------------------------------
class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "flame"

    def read(self, path: Path, *, strict: bool = False, use: str | None = None,
             **_ignored) -> tuple[Lattice, FidelityReport]:
        """Parse a FLAME GLPS deck.

        ``use`` overrides the deck's ``USE:`` line.  In strict mode the first
        LOSSY/DROPPED entry raises :class:`~lattix.fidelity.TranslationError`.
        """
        path = Path(path)
        text = path.read_text(encoding="latin-1", errors="replace")
        return self.read_text(text, rep_file=str(path), strict=strict, use=use)

    def read_text(self, text: str, *, rep_file: str | None = None, strict: bool = False,
                  use: str | None = None) -> tuple[Lattice, FidelityReport]:
        rep = FidelityReport(source_format="flame", source_file=rep_file)
        self._rep = rep
        self._file = rep_file
        self._pending_ref = None
        self._mass_number = 1
        self._charge_source = ""
        self._tags = parse_tags(text)
        statements = parse_statements(text)

        variables: dict[str, object] = {}
        globals_: dict[str, object] = {}
        global_exprs: dict[str, str] = {}
        elements: dict[str, Element] = {}
        lines: dict[str, Line] = {}
        order: list[str] = []
        root: str | None = None
        warnings: list[str] = []

        for st in statements:
            if st.kind == "assign":
                try:
                    value = evaluate(st.value, variables, line=st.line)
                except GLPSError as exc:
                    rep.lossy("FLAME_BAD_EXPRESSION", f"{st.name} = {expr_text(st.value)}: {exc}",
                              element=st.name, line=st.line)
                    warnings.append(f"line {st.line}: {exc}")
                    continue
                variables[st.name] = value
                globals_[st.name] = value
                global_exprs[st.name] = expr_text(st.value)
            elif st.kind == "use":
                root = st.type
            elif st.kind == "command":
                if st.name != "END":                              # glps_parser.cpp:481
                    raise GLPSError(f"undefined command {st.name!r}", st.line)
            elif st.kind == "func":
                if st.name != "print":                            # glps_parser.cpp:491
                    raise GLPSError(f"undefined global function {st.name!r}", st.line)
            elif st.kind == "line":
                if st.name in lines:
                    raise GLPSError(f"name {st.name!r} already used", st.line)
                lines[st.name] = Line(
                    name=st.name,
                    items=[LineItem(ref=r.name, repeat=r.repeat, reverse=r.reverse)
                           for r in _line_refs(st.items, st.line)],
                )
            else:
                self._ensure_ref(globals_, global_exprs, rep, warnings)
                params: dict[str, object] = {}
                for key, node in st.props:
                    try:
                        params[key] = evaluate(node, variables, line=st.line)
                    except GLPSError as exc:
                        # FLAME itself aborts here; permissive mode drops the attribute so the
                        # rest of the deck can still be read (FrontEnd.lat ships `ver = v`).
                        rep.lossy("FLAME_BAD_EXPRESSION",
                                  f"{st.name}.{key} = {expr_text(node)}: {exc}; attribute dropped",
                                  element=st.name, kind=st.type, line=st.line)
                        warnings.append(f"line {st.line}: {st.name}.{key}: {exc}")
                el = self._element(st.name, st.type, params, st.line, rep, warnings)
                if st.name in elements:
                    raise GLPSError(f"name {st.name!r} already used", st.line)
                elements[st.name] = el
                order.append(st.name)

        # FLAME reads IonEs/IonEk/SampleFreq from the machine-level Config, which is a
        # flat last-write-wins map, so the *final* globals define the reference; the one
        # built at the first element only supplied the rigidity for a normalized sbend K.
        self._ensure_ref(globals_, global_exprs, rep, warnings)
        reference = self._reference(globals_, global_exprs, rep, warnings, record=False)
        root = use or root
        if root is None:
            if lines:
                root = next(reversed(lines))
                warnings.append(f"deck has no 'USE:' statement; using the last line {root!r}")
            elif order:
                root = "__flame__"
                lines[root] = Line(name=root, items=[LineItem(ref=n) for n in order])
                warnings.append("deck has no LINE; every element used in definition order")
        elif root not in lines:
            raise GLPSError(f"USE: {root!r} is not a defined line")

        lat = Lattice(
            name=Path(rep_file).stem if rep_file else "flame",
            elements=elements, lines=lines, use=root, reference=reference,
            variables={k: Variable(value=float(v)) for k, v in globals_.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)},
            warnings=warnings,
        )
        lat.meta["flame"] = {
            "globals": {k: v for k, v in globals_.items() if k in GLOBAL_KEYS},
            "global_exprs": {k: v for k, v in global_exprs.items() if k in GLOBAL_KEYS},
            "sim_type": globals_.get("sim_type", "MomentMatrix"),
            "sample_freq_Hz": float(globals_.get("SampleFreq", SAMPLE_FREQ_DEFAULT)),
            "mass_number": getattr(self, "_mass_number", 1),
            "charge_expr": getattr(self, "_charge_source", ""),
            "beam_vectors": {k: v for k, v in globals_.items()
                             if isinstance(v, list) and k not in GLOBAL_KEYS},
        }
        rep.raise_if(strict)
        return lat, rep

    # -- reference particle ---------------------------------------------
    def _ensure_ref(self, g: dict, exprs: dict, rep: FidelityReport,
                    warnings: list[str]) -> ReferenceParticle:
        """Build the reference particle once, from the globals seen so far.

        FLAME needs ``IonEs``/``IonEk``/``IonChargeStates`` before the lattice, so
        the first element statement is the right moment: the normalized ``sbend K``
        then has a rigidity to be denormalized with.
        """
        if self._pending_ref is None:
            self._pending_ref = self._reference(g, exprs, rep, warnings)
        return self._pending_ref

    def _reference(self, g: dict, exprs: dict, rep: FidelityReport, warnings: list[str],
                   record: bool = True) -> ReferenceParticle:
        if not record:
            rep = FidelityReport()
            warnings = []
        mass_per_u = float(g.get("IonEs", AMU_eV))
        if "IonEs" not in g:
            rep.lossy("FLAME_NO_IONES", f"deck has no IonEs; assuming {AMU_eV:g} eV/u (AMU)")
        ek_per_u = float(g.get("IonEk", 0.0))
        if "IonEk" not in g:
            rep.lossy("FLAME_NO_IONEK", "deck has no IonEk; reference kinetic energy set to 0")

        states = g.get("IonChargeStates")
        if isinstance(states, list) and states:
            q_over_a = float(states[0])
            if len(states) > 1:
                rep.equivalent("MULTI_CHARGE_STATE_FIRST",
                               f"deck has {len(states)} charge states; the IR keeps the first "
                               f"(Q/A = {q_over_a:.9g})",
                               states=[float(s) for s in states])
        elif "IonZ" in g:
            q_over_a = float(g["IonZ"])
        else:
            q_over_a = 1.0
            rep.lossy("FLAME_NO_CHARGE_STATE",
                      "deck has neither IonChargeStates nor IonZ; assuming Q/A = 1")

        charge, mass_number, err = charge_and_mass_number(q_over_a)
        if err > 1e-9:
            rep.equivalent("FLAME_CHARGE_RATIO_APPROX",
                           f"Q/A = {q_over_a:.12g} is not a simple rational; using "
                           f"Q = {charge}, A = {mass_number} (relative error {err:.3g})",
                           q_over_a=q_over_a, charge=charge, mass_number=mass_number, error=err)
        source = exprs.get("IonChargeStates") or exprs.get("IonZ") or ""
        name = ("proton" if (charge, mass_number) == (1, 1) and abs(mass_per_u - 938.272e6) < 1e6
                else f"ion_A{mass_number}_Q{charge}")
        species = Species(name=name, mass_eV=mass_per_u * mass_number, charge=charge)
        freq = float(g.get("SampleFreq", SAMPLE_FREQ_DEFAULT))
        if "SampleFreq" not in g:
            warnings.append(f"deck has no SampleFreq; FLAME's default {SAMPLE_FREQ_DEFAULT:g} Hz "
                            "is the reference RF clock")
        self._charge_source = source
        self._mass_number = mass_number
        return ReferenceParticle(species=species, kinetic_energy_eV=ek_per_u * mass_number,
                                 rf_frequency_Hz=freq)

    # -- elements ---------------------------------------------------------
    def _element(self, name: str, type_: str, params: dict, line: int, rep: FidelityReport,
                 warnings: list[str]) -> Element:
        kind = type_.lower()
        consumed: set[str] = set()

        def num(key: str, default: float = 0.0) -> float:
            consumed.add(key)
            v = params.get(key, default)
            if isinstance(v, str) or isinstance(v, list):
                raise GLPSError(f"{name}.{key} must be a number, got {v!r}", line)
            return float(v)

        length = num("L") if "L" in params else 0.0
        consumed.add("L")
        common = {"name": name, "length": length}

        if kind in ("marker", "source", "bpm", "generic"):
            length = 0.0
            common["length"] = 0.0
        if kind == "drift":
            el: Element = Drift(**common)
        elif kind == "marker":
            el = Marker(**common)
        elif kind == "source":
            el = Marker(**common)
            for key in ("vector_variable", "matrix_variable"):
                consumed.add(key)
        elif kind == "bpm":
            el = Instrument(**common, family="BPM")
        elif kind == "quadrupole":
            el = Quadrupole(**common)
            el.multipole.Bn[1] = num("B2")
        elif kind == "sextupole":
            el = Sextupole(**common)
            el.multipole.Bn[2] = num("B3")
        elif kind == "solenoid":
            el = Solenoid(**common)
            el.solenoid.Bsol_T = num("B")
        elif kind == "orbtrim":
            el = self._orbtrim(common, params, num, consumed, rep, name)
        elif kind == "sbend":
            el = self._sbend(common, params, num, consumed, rep, name, line)
        elif kind == "rfcavity":
            el = self._rfcavity(common, params, num, consumed, rep, name)
        elif kind == "stripper":
            el = Foil(**common)
            rep.equivalent("FLAME_STRIPPER_MODEL",
                           "FLAME's Baron-formula charge stripper is kept as a Foil; its "
                           "charge-state redistribution has no IR model",
                           element=name, kind="Foil")
        elif kind == "tmatrix":
            el = self._tmatrix(common, params, consumed, rep, name, line)
        elif kind in ("edipole", "equad"):
            el = Marker(**common)
            rep.dropped("ELECTROSTATIC_UNSUPPORTED",
                        f"FLAME {kind!r} is an electrostatic element with no IR model; kept as "
                        f"a marker of length {length:g} m (parameters preserved in native)",
                        element=name, kind="Marker", line=line, flame_type=kind)
        else:
            el = Marker(**common)
            rep.dropped("FLAME_UNKNOWN_TYPE",
                        f"unknown FLAME element type {type_!r}; kept as a marker",
                        element=name, kind="Marker", line=line, flame_type=type_)
            warnings.append(f"line {line}: unknown FLAME element type {type_!r} on {name!r}")

        # aperture: `aper` is a deck convention FLAME's own code never reads
        if "aper" in params:
            consumed.add("aper")
            radius = float(params["aper"])
            if radius > 0:
                el.aperture = ApertureP.circle(radius)

        # misalignments (m / rad)
        shift = BodyShiftP(
            x_offset=num("dx"), y_offset=num("dy"),
            x_rot=num("pitch"), y_rot=num("yaw"), tilt=num("roll"),
        )
        if not shift.is_zero():
            el.shift = shift

        leftover = {k: v for k, v in params.items() if k not in consumed}
        native: dict = {"type": kind}
        if leftover:
            native["attrs"] = leftover
        el.native["flame"] = native
        tag = self._tags.get(line)
        el.provenance = Provenance(format="flame", file=self._file, line=line,
                                   original_name=tag[0] if tag else name,
                                   original_type=tag[1] if tag else kind)
        return el

    # -- per-type helpers -------------------------------------------------
    def _orbtrim(self, common, params, num, consumed, rep, name) -> Kicker:
        realpara = num("realpara") == 1.0
        theta_x, theta_y = num("theta_x"), num("theta_y")
        tm_x, tm_y = num("tm_xkick"), num("tm_ykick")
        el = Kicker(**common)
        if realpara:
            # moment.cpp:888-891: theta = tm * IonZ·c/sqrt(W²−Es²) = tm / Brho
            ref = getattr(self, "_pending_ref", None)
            if ref is not None:
                el.hkick, el.vkick = kick_from_bl(tm_x, ref), kick_from_bl(tm_y, ref)
            else:
                el.hkick, el.vkick = 0.0, 0.0
                rep.equivalent("FLAME_ORBTRIM_REALPARA",
                               "orbtrim with realpara = 1 gives ∫B·dl [T·m]; the deflection "
                               "depends on the local rigidity and is kept in native only",
                               element=name, kind="Kicker", tm_xkick=tm_x, tm_ykick=tm_y)
                consumed.discard("tm_xkick")
                consumed.discard("tm_ykick")
        else:
            # moment.cpp:897-898: transfer(PS_PX, 6) = +theta_x  → +x' per rad, MAD-X sign
            el.hkick, el.vkick = theta_x, theta_y
        if num("xyrotate") != 0.0:
            consumed.discard("xyrotate")
            rep.equivalent("FLAME_ORBTRIM_XYROTATE",
                           "orbtrim 'xyrotate' rotates the beam about s before the kick; the IR "
                           "Kicker has no rotation, the value is kept in native",
                           element=name, kind="Kicker", xyrotate_deg=params.get("xyrotate"))
        return el

    def _sbend(self, common, params, num, consumed, rep, name, line) -> Bend:
        el = Bend(**common)
        el.bend.angle = math.radians(num("phi"))
        el.bend.e1 = math.radians(num("phi1"))
        el.bend.e2 = math.radians(num("phi2"))
        if num("ver") == 1.0:
            el.bend.tilt_ref = math.pi / 2
        k1 = num("K")
        if k1:
            # moment.cpp:1020 K/sqr(MtoMM) + moment_sup.cpp:228-230 Kx = K + 1/rho², Ky = −K
            el.multipole.Bn[1] = gradient_from_k1(k1, self._ref_for_strength())
            el.meta["flame_K"] = k1
        for key in ("dphi1", "dphi2"):
            if num(key) != 0.0:
                consumed.discard(key)
                rep.equivalent("FLAME_BEND_DPHI",
                               f"sbend {key!r} is an extra pole-face angle FLAME adds on top of "
                               f"phi1/phi2; kept in native only",
                               element=name, kind="Bend", line=line)
        if "bg" in params:
            consumed.add("bg")
            el.meta["flame_bg"] = float(params["bg"])
        return el

    def _rfcavity(self, common, params, num, consumed, rep, name) -> RFCavity:
        el = RFCavity(**common)
        for key in ("cavtype", "datafile"):
            consumed.add(key)
        el.rf.frequency_Hz = num("f") or None
        syncflag = num("syncflag", 1.0)
        el.rf.phase_rad = math.radians(num("phi"))
        el.rf.phase_is_sync = syncflag != 0.0
        el.meta["flame_syncflag"] = int(syncflag)
        el.rf.L_active_m = common["length"] or None
        scl = num("scl_fac", 1.0)
        cavtype = params.get("cavtype")
        el.meta["flame_cavtype"] = cavtype
        el.meta["flame_scl_fac"] = scl
        rep.equivalent("FLAME_CAVTYPE_VOLTAGE_UNKNOWN",
                       f"FLAME rfcavity {cavtype!r} takes its accelerating field from a tabulated "
                       f"transit-time model scaled by scl_fac = {scl:g}; the equivalent voltage is "
                       "not in the deck, so RFP.voltage_V stays 0",
                       element=name, kind="RFCavity", cavtype=cavtype, scl_fac=scl)
        if syncflag == 0.0:
            rep.equivalent("FLAME_DRIVEN_PHASE",
                           "rfcavity syncflag = 0: 'phi' is a driven (entrance) RF phase, not a "
                           "synchronous phase; RFP.phase_is_sync is False",
                           element=name, kind="RFCavity")
        return el

    def _tmatrix(self, common, params, consumed, rep, name, line) -> Taylor:
        """FLAME's state is ``(x mm, x' rad, y mm, y' rad, phi rad, dEk MeV/u)``; the IR map is
        SI, so the transverse rows/columns are rescaled to metres here (``R21 → R21·1e3``,
        ``R12 → R12/1e3``: the writer does the exact inverse).  The longitudinal pair has no
        SI equivalent without the reference particle, so terms coupling to it stay in FLAME's
        units and are flagged (``EQUIVALENT:TAYLOR_BASIS_FLAME``)."""
        consumed.add("matrix")
        raw = params.get("matrix")
        el = Taylor(basis="common", **common)
        if not isinstance(raw, list) or len(raw) != 49:
            rep.lossy("FLAME_TMATRIX_SHAPE",
                      f"tmatrix 'matrix' must hold 49 numbers, got "
                      f"{len(raw) if isinstance(raw, list) else type(raw).__name__}; identity used",
                      element=name, kind="Taylor", line=line)
            return el
        m = [[float(raw[7 * i + j]) for j in range(7)] for i in range(7)]
        scale = [1e-3, 1.0, 1e-3, 1.0, 1.0, 1.0]          # mm -> m on x and y
        el.matrix = [[m[i][j] * scale[i] / scale[j] for j in range(6)] for i in range(6)]
        el.offset = [m[i][6] * scale[i] for i in range(6)]     # the 7th column is the constant kick
        el.meta["flame_matrix_row6"] = m[6]
        coupled = any(m[i][j] for i in range(4) for j in (4, 5)) or \
            any(m[i][j] for i in (4, 5) for j in range(4)) or any(m[i][6] for i in (4, 5))
        longitudinal = any(m[i][j] != (1.0 if i == j else 0.0) for i in (4, 5) for j in (4, 5))
        if coupled or longitudinal:
            rep.equivalent("TAYLOR_BASIS_FLAME",
                           "the tmatrix's longitudinal terms (phi rad, dEk MeV/u) were kept in "
                           "FLAME's units; only the transverse block is in the IR's SI basis",
                           element=name, kind="Taylor", line=line)
        return el

    # -- helpers ----------------------------------------------------------
    def _ref_for_strength(self) -> ReferenceParticle:
        """Rigidity used to turn a normalized ``K`` into a lab gradient.

        ``sbend K`` is normalized, so the lab gradient depends on the reference
        particle.  The globals are parsed before the elements in every real deck
        (FLAME needs them too), so ``_pending_ref`` is normally set; a deck that
        defines a bend before ``IonEk`` falls back to a 1 T·m rigidity and the
        gradient is then only right up to that factor.
        """
        ref = getattr(self, "_pending_ref", None)
        if ref is None:
            ref = ReferenceParticle(species=Species(name="unknown", mass_eV=AMU_eV, charge=1),
                                    kinetic_energy_eV=0.0)
        return ref


def read(path: str | Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)


__all__ = ["AMU_eV", "ELEMENT_TYPES", "GLPSError", "Node", "Reader", "SAMPLE_FREQ_DEFAULT",
           "Statement", "TAG_RE", "Token", "charge_and_mass_number", "evaluate", "expr_text",
           "parse_statements", "parse_tags", "read", "tokenize"]
