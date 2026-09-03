"""MAD8 flat-file writer (PLAN §6 task 2.3) — a dialect of the MAD-X writer.

MAD8 is MAD-X's ancestor, so the physics rules are the MAD-X writer's
(:mod:`lattix.formats.madx.writer`) and only the *spelling* differs:

=========================  ==================================  ==========================
what                       MAD-X                                MAD8
=========================  ==================================  ==========================
comments                   ``!`` (and ``//``)                   ``!``
statement end              ``;``                                end of line, ``&`` continues
layout                     ``sequence`` … ``endsequence``       ``LINE=(…)`` only
case                       lower case                           UPPER case
identifier length          41 characters (measured)             16 characters (documented)
rbend ``l``                the chord (``rbarc`` defaults true)  the **arc** (no ``rbarc``)
exit fringe integral       ``fintx``                            ``fint`` only
explicit map               ``matrix``                           none → marker/drift
particles                  many named + ``ion``                 POSITRON/ELECTRON/PROTON/
                                                                ANTI-PROTON only
=========================  ==================================  ==========================

Because MAD8 knows only four particles, an H⁻ (or any other) beam is written the way the
PIP-II decks do it — ``BEAM, MASS=…, CHARGE=…, ENERGY=…`` plus a ``BRHO := …`` parameter,
which is also what this package's reader resolves the rigidity from.  Writing ``PARTICLE=``
is reserved for the four names MAD8 accepts.

Reference energy: MAD8's ``RFCAVITY`` leaves ``p0`` alone exactly as MAD-X's does, so an
accelerating line cannot be exact.  ``energy_mode="constant"`` (the default here — MAD8 has
no MAD-X ``TWISS`` orbit-``pt`` bookkeeping to compose with) normalises every strength with
the rigidity at the start of the lattice; ``energy_mode="local"`` uses the rigidity at each
element's entrance so each section's optics is right.  Either way the choice is in the
ledger (``EQUIVALENT:CONST_P0_START_RIGIDITY`` / ``…_LOCAL_RIGIDITY``), never silent.

Numbers are ``%.15g``; names are sanitized to upper case, capped at 16 characters and made
unique, with a ``! lattix: name="…" type="…"`` comment line above the definition that
:func:`lattix.formats.mad8.reader.parse_tags` reads back (invariant I-15).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    ALL_KINDS,
    ApertureP,
    Directive,
    Element,
    Freq,
    Superposition,
)
from lattix.ir.expr import ExpressionError, evaluate
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import ReferenceParticle
from lattix.ir.rf import madx_lag
from lattix.ir.walk import energy_gain_eV, propagate

#: MAD8 identifiers: a letter followed by letters, digits, ``_`` and ``.``; the
#: documented limit is 16 characters (the PIP-II decks stay at or below it).
NAME_RE = re.compile(r"^[A-Z][A-Z0-9_.]*$")
MAX_NAME_LEN = 16

#: the four particles MAD8 knows by name.
NAMED_PARTICLES: dict[str, str] = {"positron": "POSITRON", "electron": "ELECTRON",
                                   "proton": "PROTON", "antiproton": "ANTI-PROTON"}

#: MAD8 element keywords: a definition ``X: DRIFT`` is parsed by keyword, and MAD8 lets an
#: element *class* stand in for a type, so a magnet or line named after a base type would
#: be ambiguous.  Command names are deliberately NOT here — ``CELL: LINE=(…)`` is the most
#: common line name in MAD8 decks and parses unambiguously (the ``name:`` prefix decides).
RESERVED: frozenset[str] = frozenset({
    "DRIFT", "SBEND", "RBEND", "QUADRUPOLE", "SEXTUPOLE", "OCTUPOLE", "MULTIPOLE",
    "SOLENOID", "RFCAVITY", "ELSEPARATOR", "KICKER", "HKICKER", "VKICKER", "MONITOR",
    "HMONITOR", "VMONITOR", "INSTRUMENT", "ECOLLIMATOR", "RCOLLIMATOR", "MARKER",
    "SROT", "YROT", "BEAMBEAM", "LUMP", "MATRIX", "LINE", "LIST", "SEQUENCE",
})

#: names a *parameter* must not take: a bare ``USE = 3`` would be read back as a command,
#: and the built-in constants would be shadowed.
RESERVED_PARAMETERS: frozenset[str] = RESERVED | frozenset({
    "BEAM", "USE", "TITLE", "RETURN", "STOP", "CALL", "SAVE", "SAVELINE", "SELECT",
    "TWISS", "SURVEY", "MATCH", "ENDMATCH", "CELL", "VARY", "PRINT", "PLOT", "OPTION",
    "VALUE", "SHOW", "TRACK", "ENDTRACK", "EALIGN", "SET", "TRUE", "FALSE", "PI",
    "TWOPI", "CLIGHT", "EMASS", "PMASS", "DEGRAD", "RADDEG",
})

#: IR ``Instrument.family`` → MAD8 base type (used when ``native["mad8"]["type"]`` is
#: absent, e.g. for a lattice that came from another format).
FAMILY_TYPE = {"BPM": "MONITOR", "MONITOR": "MONITOR", "HMONITOR": "HMONITOR",
               "VMONITOR": "VMONITOR", "INSTRUMENT": "INSTRUMENT",
               "PLACEHOLDER": "INSTRUMENT", "BLM": "BLMONITOR", "SLM": "SLMONITOR",
               "WIRE": "WIRE", "PROFILE": "PROFILE"}

#: MAD8 base types that take no ``APERTURE`` attribute.
NO_APERTURE = frozenset({"MARKER", "SROT", "YROT"})

#: an element counts as accelerating above 1 µeV of reference gain (MAD-X writer rule).
_ACCEL_TOL_eV = 1e-6
_EXPR_TOL = 1e-12
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
#: MAD8 has no line-length limit in practice, but the 80-column card is the convention
#: every deck in the corpus follows; longer statements are broken with ``&``.
WRAP_COLUMNS = 78


def sanitize(name: str) -> str:
    """Upper-case *name* and replace everything MAD8 cannot spell with ``_``."""
    s = re.sub(r"[^A-Z0-9_.]", "_", (name or "").strip().upper())
    s = s.lstrip("_.")
    if not s or not s[0].isalpha():
        s = "E_" + s
    if len(s) > MAX_NAME_LEN:
        s = s[:MAX_NAME_LEN]
    if s in RESERVED:
        s = s[:MAX_NAME_LEN - 2] + "_X"
    return s


def is_valid(name: str) -> bool:
    """True when *name* can be a MAD8 parameter (the strictest of the two name rules)."""
    return (bool(NAME_RE.match(name)) and len(name) <= MAX_NAME_LEN
            and name not in RESERVED_PARAMETERS)


class NameMap:
    """Sanitize + uniquify names, remembering what was renamed (PLAN §4.4)."""

    def __init__(self) -> None:
        self._assigned: dict[int, str] = {}
        self._by_original: dict[str, str] = {}
        self._used: set[str] = set()
        self.renamed: dict[str, str] = {}

    def assign(self, original: str, key: object | None = None) -> str:
        cache_key = id(key) if key is not None else None
        if cache_key is not None and cache_key in self._assigned:
            return self._assigned[cache_key]
        if cache_key is None and original in self._by_original:
            return self._by_original[original]
        name = self._unique(sanitize(original))
        if cache_key is not None:
            self._assigned[cache_key] = name
        self._by_original[original] = name
        if name != original:
            self.renamed[name] = original
        return name

    def reserve(self, name: str) -> str:
        return self._unique(sanitize(name))

    def _unique(self, base: str) -> str:
        out = base
        k = 2
        while out in self._used:
            suffix = f"_{k}"
            out = base[:MAX_NAME_LEN - len(suffix)] + suffix
            k += 1
        self._used.add(out)
        return out


def name_tag(original: str, original_type: str | None = None) -> str:
    body = f'name="{original}"'
    if original_type:
        body += f' type="{original_type}"'
    return f"! lattix: {body}"


def _expr_text(text: str) -> str:
    """MAD8 is case insensitive and its decks are upper case; an expression imported from
    a lower-case dialect (MAD-X) is upper-cased so the deck reads consistently."""
    return str(text).strip().upper()


def _num(x: float) -> str:
    s = f"{float(x):.15g}"
    return "0" if s in ("-0", "-0.0") else s


def wrap(statement: str, columns: int = WRAP_COLUMNS) -> list[str]:
    """Break a statement across MAD8 ``&`` continuation cards, preferring commas."""
    if len(statement) <= columns:
        return [statement]
    out: list[str] = []
    rest = statement
    while len(rest) > columns:
        cut = rest.rfind(",", 0, columns)
        cut = cut + 1 if cut > 0 else columns - 1
        out.append(rest[:cut] + "&")
        rest = rest[cut:].lstrip()
    out.append(rest)
    return out


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str
    cls: str = "EXACT"
    code: str = "OK"
    message: str = ""


@dataclass
class _Attr:
    """One MAD8 attribute of one element definition."""

    key: str
    value: float | None = None
    path: str | None = None        # IR expression path (``multipole.Bn[1]``)
    text: str | None = None        # verbatim rendering (integers, flags)


@dataclass
class _Item:
    element: Element
    s_in: float
    s_out: float
    brho: float
    dE: float


@dataclass
class _Ctx:
    """What the expression check needs: variable values and element attribute values."""

    variables: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, dict[str, float]] = field(default_factory=dict)

    def resolve(self, token: str) -> float:
        if "[" in token and token.endswith("]"):
            head, _, tail = token.partition("[")
            table = self.attrs.get(head.upper())
            if table is None:
                raise ExpressionError(f"unknown element {head!r}")
            key = tail[:-1].lower()
            if key not in table:
                raise ExpressionError(f"{token} is not written by this deck")
            return table[key]
        raise ExpressionError(f"unknown identifier {token!r}")


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`."""

    format = "mad8"

    RULES: dict[str, Rule] = {
        "Drift": Rule("DRIFT"),
        "Quadrupole": Rule("QUADRUPOLE"),
        "Sextupole": Rule("SEXTUPOLE"),
        "Octupole": Rule("OCTUPOLE"),
        "Multipole": Rule("MULTIPOLE"),
        "Bend": Rule("SBEND/RBEND"),
        "Solenoid": Rule("SOLENOID"),
        "RFCavity": Rule("RFCAVITY"),
        "FieldMap": Rule("DRIFT", "LOSSY", "FM_TO_DRIFT",
                         "field map replaced by a drift of the same length"),
        "NCells": Rule("DRIFT", "LOSSY", "NCELLS_TO_DRIFT",
                       "NCELLS cell train replaced by a drift of the same length"),
        "RFQCell": Rule("DRIFT", "LOSSY", "RFQ_TO_DRIFT",
                        "RFQ cell replaced by a drift of the same length"),
        "Kicker": Rule("KICKER/HKICKER/VKICKER"),
        "Collimator": Rule("RCOLLIMATOR/ECOLLIMATOR"),
        "Marker": Rule("MARKER"),
        "Instrument": Rule("MONITOR/HMONITOR/VMONITOR/INSTRUMENT"),
        "Foil": Rule("MARKER", "LOSSY", "FOIL_TO_MARKER",
                     "MAD8 has no stripping foil; written as a marker"),
        "Taylor": Rule("MARKER/DRIFT", "LOSSY", "TAYLOR_DROPPED",
                       "MAD8 has no MATRIX element; the map is written as a marker "
                       "(or a drift of the same length)"),
        "Patch": Rule("MARKER", "LOSSY", "PATCH_DROPPED",
                      "MAD8 has no patch element; written as a marker"),
        "ReferenceChange": Rule("MARKER", "LOSSY", "REFCHANGE_DROPPED",
                                "MAD8 cannot change the reference energy; written as a marker"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each MAD8 cavity's FREQ attribute"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                              "overlapping fields written as consecutive elements"),
    }

    #: Directive roles a comment can carry without losing physics.
    COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              energy_mode: str = "constant", use_expressions: bool = True,
              line_name: str | None = None) -> FidelityReport:
        if energy_mode not in ("local", "constant"):
            raise ValueError(f"energy_mode must be 'local' or 'constant', got {energy_mode!r}")

        rep = FidelityReport(target_format="mad8", target_file=str(path))
        placed = propagate(lattice)
        items = self._items(lattice, placed, rep)
        self._record_energy_mode(items, energy_mode, rep)

        names = NameMap()
        root = lattice.use or lattice.name
        root_name = names.reserve(line_name or (root if root in lattice.lines else None)
                                  or lattice.name or root or "LATTIX")
        line_names: dict[str, str] = {}
        if root in lattice.lines:
            line_names[root] = root_name
            for ln in lattice.lines:
                if ln != root:
                    line_names[ln] = names.reserve(ln)

        start_brho = lattice.reference.brho_signed
        defs: dict[int, tuple[str, Element, float]] = {}
        order: list[int] = []
        for it in items:
            key = id(it.element)
            if key in defs:
                brho0 = defs[key][2]
                local = it.brho if energy_mode == "local" else start_brho
                if abs(local - brho0) > 1e-12 * max(1.0, abs(brho0)):
                    rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                                   "one definition is used at two reference energies; the first "
                                   "occurrence's rigidity is used for its normalized strengths",
                                   element=it.element.name, kind=it.element.kind,
                                   brho_first=brho0, brho_here=local)
                continue
            defs[key] = (names.assign(it.element.name, it.element), it.element,
                         it.brho if energy_mode == "local" else start_brho)
            order.append(key)

        # pass 1: every definition's MAD8 attributes (one call per element, so the
        # per-kind builders record their ledger entries exactly once)
        built: dict[int, tuple[str, list[_Attr]]] = {}
        ctx = _Ctx(variables=self._variable_values(lattice))
        for key in order:
            name, el, brho = defs[key]
            base, attrs = self._build(el, brho, rep)
            built[key] = (base, attrs)
            ctx.attrs[name] = {a.key: a.value for a in attrs if a.value is not None}

        body: list[str] = [f"! lattix {__version__} from {lattice.meta.get('source_format', 'IR')}"]
        title = (lattice.meta.get("mad8_title") or lattice.meta.get("madx_title")
                 or lattice.meta.get("title"))
        if title:
            body.append(f'TITLE, "{title}"')
        body.append("")
        body.extend(self._beam_lines(lattice.reference, rep))
        body.append("")
        body.append("! --- parameters")
        body.extend(self._parameter_lines(lattice, use_expressions, rep))

        body.append("")
        body.append("! --- elements")
        for key in order:
            name, el, brho = defs[key]
            text = self._definition(name, el, built[key], ctx, use_expressions, rep)
            if text:
                body.extend(text)

        body.append("")
        body.append("! --- lines")
        body.extend(self._line_block(root_name, line_names, lattice, defs, rep))
        body.append("")
        body.append(f"USE, {root_name}")
        body.append("RETURN")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(body).rstrip("\n") + "\n", encoding="latin-1",
                              errors="replace")
        rep.raise_if(strict)
        return rep

    # -- flattening ---------------------------------------------------------
    def _items(self, lattice: Lattice, placed: list[Placed], rep: FidelityReport) -> list[_Item]:
        out: list[_Item] = []
        for p in placed:
            el = p.element
            ref: ReferenceParticle = p.ref_in or lattice.reference
            if isinstance(el, Superposition):
                rep.lossy("SUPERPOSITION_FLATTENED",
                          "overlapping fields written as consecutive MAD8 elements",
                          element=el.name, kind="Superposition", children=len(el.children))
                for offset, child_name in el.children:
                    child = lattice.elements.get(child_name)
                    if child is None:
                        rep.dropped("SUPERPOSITION_CHILD_MISSING",
                                    f"superposition child {child_name!r} is not defined",
                                    element=el.name, kind="Superposition")
                        continue
                    s0 = p.s_in + offset
                    out.append(_Item(child, s0, s0 + child.length, ref.brho_signed, 0.0))
                continue
            out.append(_Item(el, p.s_in, p.s_out, ref.brho_signed, energy_gain_eV(el, ref)))
        return out

    @staticmethod
    def _record_energy_mode(items: list[_Item], energy_mode: str, rep: FidelityReport) -> None:
        local = energy_mode == "local"
        code = "CONST_P0_LOCAL_RIGIDITY" if local else "CONST_P0_START_RIGIDITY"
        msg = ("normalized strengths use the local rigidity at each element's entrance"
               if local else "normalized strengths use the rigidity at the start of the lattice")
        for it in items:
            if abs(it.dE) <= _ACCEL_TOL_eV:
                continue
            rep.equivalent("CONST_P0",
                           "MAD8 keeps p0 constant across RF: the reference energy does not "
                           "follow this element's gain",
                           element=it.element.name, kind=it.element.kind, dE_eV=it.dE)
            rep.equivalent(code, msg, element=it.element.name, kind=it.element.kind,
                           dE_eV=it.dE, brho=it.brho)

    # -- header ---------------------------------------------------------------
    @staticmethod
    def _beam_lines(ref: ReferenceParticle, rep: FidelityReport) -> list[str]:
        sp = ref.species
        parts = []
        named = NAMED_PARTICLES.get(sp.name.lower())
        if named:
            parts.append(f"PARTICLE={named}")
        else:
            rep.equivalent("BEAM_MASS_CHARGE",
                           f"MAD8 knows only {sorted(NAMED_PARTICLES.values())}; {sp.name!r} is "
                           "written as MASS/CHARGE plus a BRHO parameter (the PIP-II convention)",
                           element=None, kind=None, species=sp.name,
                           mass_GeV=sp.mass_eV / 1e9, charge=sp.charge)
        parts.append(f"MASS={_num(sp.mass_eV / 1e9)}")
        parts.append(f"CHARGE={_num(sp.charge)}")
        parts.append(f"ENERGY={_num(ref.total_energy_eV / 1e9)}")
        return wrap("BEAM, " + ", ".join(parts))

    @staticmethod
    def _variable_values(lattice: Lattice) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, var in lattice.variables.items():
            out[name] = var.value
            out[name.lower()] = var.value
        return out

    def _parameter_lines(self, lattice: Lattice, use_expressions: bool,
                         rep: FidelityReport) -> list[str]:
        """``BRHO := …`` (the reader's rigidity source) plus the lattice's variables.

        Parameters the reader could not turn into usable variables (MAD8 spells a
        chromaticity ``QX'``, which no expression parser can reference) ride along in
        ``meta["mad8_unparseable_params"]`` and are written back verbatim, so a MAD8 round
        trip keeps them.
        """
        brho = abs(lattice.reference.brho_signed)
        lines: list[str] = []
        keep: dict[str, object] = {}
        own_brho = True
        if use_expressions:
            for name, var in lattice.variables.items():
                upper = name.upper()
                if not is_valid(upper) and upper != "BRHO":
                    rep.equivalent("VARIABLE_DROPPED",
                                   f"variable {name!r} is not a writable MAD8 identifier; its "
                                   "value is folded into the numbers",
                                   element=None, kind=None, variable=name)
                    continue
                if upper == "BRHO":
                    if abs(var.value - brho) <= 1e-9 * max(brho, 1.0):
                        own_brho = False            # the deck's own BRHO already agrees
                    else:
                        rep.equivalent("BRHO_REPLACED",
                                       f"the lattice's BRHO variable ({var.value:.6g}) disagrees "
                                       f"with the reference rigidity ({brho:.6g} T·m); the "
                                       "rigidity wins so the strengths stay consistent",
                                       element=None, kind=None, brho_variable=var.value,
                                       brho_reference=brho)
                        continue
                keep[upper] = var
        if own_brho:
            lines.append(f"BRHO := {_num(brho)}")
        for name in self._topo(keep):
            var = keep[name]
            expr = getattr(var, "expression", None)
            if expr is not None and expr.text.strip():
                lines.extend(wrap(f"{name} := {_expr_text(expr.text)}"))
            else:
                lines.append(f"{name} := {_num(var.value)}")
        if use_expressions:
            for name, text in (lattice.meta.get("mad8_unparseable_params") or {}).items():
                lines.extend(wrap(f"{name} := {str(text).strip()}"))
        return lines

    @staticmethod
    def _topo(variables: dict) -> list[str]:
        """Definition order with dependencies first (stable otherwise)."""
        deps = {name: {t.upper() for t in _ID.findall(
                    variables[name].expression.text if variables[name].expression else "")
                    if t.upper() in variables and t.upper() != name}
                for name in variables}
        out: list[str] = []
        done: set[str] = set()

        def visit(n: str, stack: frozenset[str]) -> None:
            if n in done or n in stack:
                return
            for d in sorted(deps[n]):
                visit(d, stack | {n})
            done.add(n)
            out.append(n)

        for name in variables:
            visit(name, frozenset())
        return out

    # -- element definitions --------------------------------------------------
    def _definition(self, name: str, el: Element, built: tuple[str, list[_Attr]], ctx: _Ctx,
                    use_expressions: bool, rep: FidelityReport) -> list[str]:
        rule = self.RULES[el.kind]
        if isinstance(el, Freq):
            rep.exact(el.name, "Freq", message=rule.message)
            return []
        if isinstance(el, Directive):
            if el.role in self.COMMENT_ROLES:
                rep.exact(el.name, "Directive", code="DIRECTIVE_AS_COMMENT",
                          message=f"role {el.role!r} carries no MAD8 physics; written as a comment")
            else:
                rep.dropped(rule.code, rule.message, element=el.name, kind="Directive",
                            card=el.card, role=el.role)
            args = " ".join(el.args)
            return [f"! lattix directive: {el.card} {args}".rstrip()]

        base, attrs = built
        attrs = list(attrs) + self._aperture_attrs(el, base, rep)
        self._record(rep, el, rule)
        self._record_shift(el, rep)
        rendered = [self._render(el, a, ctx, use_expressions, rep) for a in attrs]
        joined = ", ".join(a for a in rendered if a)
        out: list[str] = []
        original = (el.provenance.original_name if el.provenance else None) or el.name
        # MAD8 is case insensitive, so upper-casing is not a rename worth tagging.
        if name != original.upper():
            out.append(name_tag(original, el.provenance.original_type if el.provenance else None))
        out.extend(wrap(f"{name}: {base}" + (f", {joined}" if joined else "")))
        return out

    @staticmethod
    def _record(rep: FidelityReport, el: Element, rule: Rule) -> None:
        if rule.cls == "EXACT":
            rep.exact(el.name, el.kind)
        else:
            rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind,
                    target=rule.target)

    @staticmethod
    def _record_shift(el: Element, rep: FidelityReport) -> None:
        """MAD8's error module is a separate command language this writer does not emit
        (there is no MAD8 binary here to verify its ``EALIGN`` spelling against), so a
        body shift is reported rather than guessed at."""
        s = el.shift
        if s is None or s.is_zero():
            return
        rep.lossy("MISALIGN_DROPPED",
                  "MAD8 alignment errors live in a separate EALIGN command block that this "
                  "writer does not emit; the offsets are recorded here only",
                  element=el.name, kind=el.kind, dx=s.x_offset, dy=s.y_offset, ds=s.z_offset,
                  dphi=s.x_rot, dtheta=s.y_rot, dpsi=s.tilt)

    @staticmethod
    def _aperture_attrs(el: Element, base: str, rep: FidelityReport) -> list[_Attr]:
        ap: ApertureP | None = el.aperture
        if ap is None or el.kind == "Collimator" or (ap.half_x is None and ap.half_y is None):
            return []
        hx = ap.half_x if ap.half_x is not None else ap.half_y
        hy = ap.half_y if ap.half_y is not None else ap.half_x
        if base in NO_APERTURE:
            rep.lossy("APERTURE_DROPPED", f"MAD8 {base} takes no APERTURE attribute",
                      element=el.name, kind=el.kind, half_x=hx, half_y=hy)
            return []
        if ap.shape != "ELLIPTICAL" or abs(hx - hy) > 1e-15:
            rep.lossy("APERTURE_APPROXIMATED",
                      "MAD8's APERTURE is a single circular half-aperture; the inscribed "
                      "radius is written",
                      element=el.name, kind=el.kind, shape=ap.shape, half_x=hx, half_y=hy)
        return [_Attr("aperture", value=min(hx, hy))]

    def _render(self, el: Element, a: _Attr, ctx: _Ctx, use_expressions: bool,
                rep: FidelityReport) -> str:
        if a.text is not None:
            return f"{a.key.upper()}={a.text}"
        if a.value is None:                     # a bare MAD8 flag
            return a.key.upper()
        if use_expressions:
            text = el.native.get("mad8", {}).get(f"{a.key}_expr")
            expr = el.expressions.get(a.path) if a.path else None
            if text is None and expr is not None:
                text = expr.text
            if text:
                try:
                    got = evaluate(str(text), ctx.variables, ctx.resolve)
                except ExpressionError:
                    got = None
                if got is not None and abs(got - a.value) <= _EXPR_TOL * max(1.0, abs(a.value)):
                    return f"{a.key.upper()}={_expr_text(text)}"
                why = ("cannot be evaluated against this deck's parameters"
                       if got is None else "no longer matches the IR value")
                rep.equivalent("EXPRESSION_DROPPED",
                               f"{a.key.upper()} expression {str(text)!r} {why}; the number is "
                               "written instead",
                               element=el.name, kind=el.kind, attr=a.key, value=a.value)
        return f"{a.key.upper()}={_num(a.value)}"

    # -- per-kind builders: (MAD8 base type, attributes) ----------------------
    def _build(self, el: Element, brho: float, rep: FidelityReport) -> tuple[str, list[_Attr]]:
        if isinstance(el, (Freq, Directive)):
            return "", []
        return getattr(self, f"_def_{el.kind.lower()}")(el, brho, rep)

    @staticmethod
    def _native_type(el: Element, allowed: set[str], default: str) -> str:
        native = str(el.native.get("mad8", {}).get("type", "")).upper()
        return native if native in allowed else default

    def _def_drift(self, el, brho, rep):
        return "DRIFT", [_Attr("l", el.length, "length")]

    def _def_quadrupole(self, el, brho, rep):
        attrs = [_Attr("l", el.length, "length"),
                 _Attr("k1", el.multipole.Bn.get(1, 0.0) / brho, "multipole.Bn[1]")]
        if el.multipole.tilt.get(1):
            attrs.append(_Attr("tilt", el.multipole.tilt[1], "multipole.tilt[1]"))
        return "QUADRUPOLE", attrs

    def _def_sextupole(self, el, brho, rep):
        return "SEXTUPOLE", self._thick_multipole(el, brho, rep, 2, "k2")

    def _def_octupole(self, el, brho, rep):
        return "OCTUPOLE", self._thick_multipole(el, brho, rep, 3, "k3")

    def _thick_multipole(self, el, brho, rep, order, key):
        attrs = [_Attr("l", el.length, "length"),
                 _Attr(key, el.multipole.Bn.get(order, 0.0) / brho, f"multipole.Bn[{order}]")]
        if el.multipole.Bs.get(order):
            rep.lossy("SKEW_COMPONENT_DROPPED",
                      f"MAD8's {key.upper()} magnet has no skew attribute; Bs[{order}] is dropped "
                      "(write it as a rotated element or a MULTIPOLE instead)",
                      element=el.name, kind=el.kind, order=order, Bs=el.multipole.Bs[order])
        if el.multipole.tilt.get(order):
            attrs.append(_Attr("tilt", el.multipole.tilt[order], f"multipole.tilt[{order}]"))
        return attrs

    def _def_multipole(self, el, brho, rep):
        """MAD8 spells a thin multipole ``K0L … K20L`` with per-order tilts ``T0 … T20``."""
        attrs: list[_Attr] = []
        lrad = el.native.get("mad8", {}).get("lrad")
        if lrad:
            attrs.append(_Attr("lrad", float(lrad)))
        for order in sorted(set(el.multipole.BnL) | set(el.multipole.BsL)):
            normal = el.multipole.BnL.get(order, 0.0) / brho
            skew = el.multipole.BsL.get(order, 0.0) / brho
            if skew and not normal:
                # a pure skew 2(n+1)-pole is the normal one rotated by π/(2(n+1))
                attrs.append(_Attr(f"k{order}l", skew))
                attrs.append(_Attr(f"t{order}", math.pi / (2 * (order + 1))))
                continue
            if normal:
                attrs.append(_Attr(f"k{order}l", normal))
            if skew:
                strength = math.hypot(normal, skew)
                attrs[-1] = _Attr(f"k{order}l", strength)
                attrs.append(_Attr(f"t{order}",
                                   -math.atan2(skew, normal) / (order + 1)))
                rep.equivalent("SKEW_MULTIPOLE_AS_TILT",
                               f"K{order}L/K{order}SL folded into one rotated normal multipole",
                               element=el.name, kind=el.kind, order=order,
                               BnL=el.multipole.BnL.get(order, 0.0), BsL=skew * brho)
            if el.multipole.tilt.get(order) and not skew:
                attrs.append(_Attr(f"t{order}", el.multipole.tilt[order]))
        return "MULTIPOLE", attrs

    def _def_bend(self, el, brho, rep):
        b = el.bend
        base = "RBEND" if b.rect else "SBEND"
        half = b.angle / 2.0 if b.rect else 0.0
        # MAD8 has no OPTION, RBARC: L is the arc length for RBEND as well as SBEND.
        attrs = [_Attr("l", el.length, "length"), _Attr("angle", b.angle, "bend.angle")]
        if (b.e1 - half) or (b.e2 - half):
            attrs.append(_Attr("e1", b.e1 - half, "bend.e1"))
            attrs.append(_Attr("e2", b.e2 - half, "bend.e2"))
        if b.edge_int1:
            attrs.append(_Attr("fint", b.edge_int1))
        if b.edge_int2 is not None and abs(b.edge_int2 - b.edge_int1) > 1e-15:
            rep.lossy("FINTX_DROPPED",
                      f"MAD8 has no FINTX: the exit fringe integral {b.edge_int2!r} is replaced "
                      f"by the entrance one ({b.edge_int1!r})",
                      element=el.name, kind="Bend", fint=b.edge_int1, fintx=b.edge_int2)
        if b.hgap:
            attrs.append(_Attr("hgap", b.hgap))
        for order, key in ((1, "k1"), (2, "k2"), (3, "k3")):
            if el.multipole.Bn.get(order):
                attrs.append(_Attr(key, el.multipole.Bn[order] / brho, f"multipole.Bn[{order}]"))
        if b.tilt_ref:
            attrs.append(_Attr("tilt", b.tilt_ref))
        return base, attrs

    def _def_solenoid(self, el, brho, rep):
        return "SOLENOID", [_Attr("l", el.length, "length"),
                            _Attr("ks", el.solenoid.Bsol_T / brho, "solenoid.Bsol_T")]

    def _def_rfcavity(self, el, brho, rep):
        rf = el.rf
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        attrs = [_Attr("l", el.length, "length"),
                 _Attr("volt", volt * 1e-6, "rf.voltage_V"),
                 _Attr("lag", madx_lag(rf.phase_rad), "rf.phase_rad")]
        if rf.frequency_Hz:
            attrs.append(_Attr("freq", rf.frequency_Hz * 1e-6, "rf.frequency_Hz"))
        harmon = el.native.get("mad8", {}).get("harmon")
        if harmon:
            attrs.append(_Attr("harmon", text=str(int(harmon))))
        return "RFCAVITY", attrs

    def _def_fieldmap(self, el, brho, rep):
        return "DRIFT", [_Attr("l", el.length)]

    _def_ncells = _def_fieldmap
    _def_rfqcell = _def_fieldmap

    def _def_kicker(self, el, brho, rep):
        if el.electric:
            rep.lossy("EKICK_AS_MAGNETIC",
                      "electric steerer written as a magnetic MAD8 kicker",
                      element=el.name, kind="Kicker")
        base = self._native_type(el, {"KICKER", "HKICKER", "VKICKER"},
                                 "HKICKER" if (el.hkick and not el.vkick) else
                                 "VKICKER" if (el.vkick and not el.hkick) else "KICKER")
        attrs = [_Attr("l", el.length, "length")]
        if base == "HKICKER":
            attrs.append(_Attr("kick", el.hkick, "hkick"))
            if el.vkick:
                rep.lossy("KICK_COMPONENT_DROPPED",
                          "an HKICKER cannot carry a vertical kick", element=el.name,
                          kind="Kicker", vkick=el.vkick)
        elif base == "VKICKER":
            attrs.append(_Attr("kick", el.vkick, "vkick"))
            if el.hkick:
                rep.lossy("KICK_COMPONENT_DROPPED",
                          "a VKICKER cannot carry a horizontal kick", element=el.name,
                          kind="Kicker", hkick=el.hkick)
        else:
            attrs.append(_Attr("hkick", el.hkick, "hkick"))
            attrs.append(_Attr("vkick", el.vkick, "vkick"))
        return base, attrs

    def _def_collimator(self, el, brho, rep):
        ap = el.aperture
        default = "RCOLLIMATOR" if (ap is not None and ap.shape == "RECTANGULAR") else "ECOLLIMATOR"
        base = self._native_type(el, {"RCOLLIMATOR", "ECOLLIMATOR"}, default)
        attrs = [_Attr("l", el.length, "length")]
        if ap is not None and ap.half_x is not None:
            attrs.append(_Attr("xsize", ap.half_x))
            attrs.append(_Attr("ysize", ap.half_y if ap.half_y is not None else ap.half_x))
        return base, attrs

    def _def_marker(self, el, brho, rep):
        if el.length:
            rep.lossy("MARKER_LENGTH_AS_DRIFT",
                      "a MAD8 MARKER has no length; the element is written as a drift",
                      element=el.name, kind=el.kind, length=el.length)
            return "DRIFT", [_Attr("l", el.length, "length")]
        return "MARKER", []

    def _def_instrument(self, el, brho, rep):
        default = FAMILY_TYPE.get(el.family.upper(), "INSTRUMENT")
        base = self._native_type(
            el, {"MONITOR", "HMONITOR", "VMONITOR", "INSTRUMENT", "BLMONITOR",
                 "SLMONITOR", "WIRE", "PROFILE"}, default)
        return base, [_Attr("l", el.length, "length")]

    def _def_foil(self, el, brho, rep):
        return self._thin_placeholder(el, rep)

    def _def_taylor(self, el, brho, rep):
        return self._thin_placeholder(el, rep)

    def _def_patch(self, el, brho, rep):
        return self._thin_placeholder(el, rep)

    def _def_referencechange(self, el, brho, rep):
        return self._thin_placeholder(el, rep)

    @staticmethod
    def _thin_placeholder(el, rep) -> tuple[str, list[_Attr]]:
        if el.length:
            return "DRIFT", [_Attr("l", el.length)]
        return "MARKER", []

    def _def_superposition(self, el, brho, rep):     # pragma: no cover - expanded in _items
        return "MARKER", []

    # -- lines -----------------------------------------------------------------
    def _line_block(self, root_name: str, line_names: dict[str, str], lattice: Lattice,
                    defs: dict, rep: FidelityReport) -> list[str]:
        by_name = {el.name: nm for nm, el, _ in defs.values()}
        root = lattice.use or lattice.name
        if root not in lattice.lines:
            rep.dropped("NO_LINE_STRUCTURE",
                        f"the lattice has no lines[{root!r}]; a flat LINE was written instead")
            flat = ", ".join(by_name[p.element.name] for p in lattice.flatten())
            return wrap(f"{root_name}: LINE=({flat})")

        sup_lines: list[str] = []
        nil: list[str] = []

        def emits_nothing(ref: str) -> bool:
            return isinstance(lattice.elements.get(ref), (Freq, Directive))

        def nil_marker() -> str:
            if not nil:
                nil.append(sanitize(f"{root_name}_NIL"))
            return nil[0]

        def ref_name(ref: str) -> str:
            if ref in line_names:
                return line_names[ref]
            if ref in by_name:
                return by_name[ref]
            el = lattice.elements.get(ref)
            if isinstance(el, Superposition):     # children were emitted, the parent was not
                nm = sanitize(ref)
                body = ", ".join(by_name[c] for _, c in el.children if c in by_name)
                sup_lines.extend(wrap(f"{nm}: LINE=({body})"))
                return nm
            return sanitize(ref)

        def item_text(it) -> str:
            txt = ref_name(it.ref)
            if it.reverse:
                txt = f"-{txt}"
            if it.repeat != 1:
                txt = f"{it.repeat}*{txt}"
            return txt

        emitted: list[str] = []
        done: set[str] = set()

        def emit(name: str) -> None:
            if name in done:
                return
            done.add(name)
            for it in lattice.lines[name].items:
                if it.ref in lattice.lines:
                    emit(it.ref)
            kept = [it for it in lattice.lines[name].items if not emits_nothing(it.ref)]
            body = ", ".join(item_text(it) for it in kept) or nil_marker()
            emitted.extend(wrap(f"{line_names.get(name, sanitize(name))}: LINE=({body})"))

        emit(root)
        head = [f"{nil[0]}: MARKER"] if nil else []
        return head + sup_lines + emitted


_MISSING = set(ALL_KINDS) - set(Writer.RULES)
if _MISSING:                                        # pragma: no cover - guarded by a test too
    raise RuntimeError(f"MAD8 writer RULES do not cover {sorted(_MISSING)}")


def write(lattice: Lattice, path: str | Path, **options) -> FidelityReport:
    return Writer().write(lattice, Path(path), **options)


__all__ = ["MAX_NAME_LEN", "RESERVED", "RESERVED_PARAMETERS", "NameMap", "Rule", "Writer",
           "is_valid", "name_tag", "sanitize", "wrap", "write"]
