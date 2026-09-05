"""Elegant ``.lte`` reader (PLAN §6 task 2.1).

Ported from HELIX ``linac_gen/io/elegant_parser.py`` (tokenizer, RPN store lines,
element templates, ``name[prop]=`` overrides, ``line=(…)`` expansion) and then
pinned against the real engine — every convention below was **measured** with
``elegant 2026.3.0`` (conda-forge, osx-arm64) on 2026-09-03, not recalled:

``RBEN``/``RBEND`` ``L`` is the **chord**
    ``B: RBEN, L=1, ANGLE=0.1`` comes back out of elegant's own ``&save_lattice``
    as ``B: SBEN, L=1.000416788226488, ANGLE=0.1, E1=0.05, E2=0.05``, i.e.
    elegant converts to ``L_arc = L·(θ/2)/sin(θ/2)`` and adds ``θ/2`` to both
    pole faces — the same rule as MAD-X ``rbarc``.  (cheetah's and ocelot's
    Elegant converters both pass ``L`` through unchanged and are wrong by
    ``L·θ²/24``; ``elegant_to_bmad.py`` inherits the error.)  The IR always
    stores the **arc** length, so the reader converts.

``FINT`` defaults to **0.5**, not 0.45
    measured from elegant's own parameter table for ``SBEN``/``CSBEND``;
    ``FINT1``/``FINT2`` default to ``-1``, meaning "use ``FINT``".

``RFCA`` phase is charge-signed
    crest is ``+90°`` for negative species and ``−90°`` for positive ones
    (:func:`lattix.ir.rf.phase_from_elegant_deg`).  Measured: a 1 MV cavity at
    ``PHASE=-120`` gives a proton ``+866025.4 eV`` and an H⁻ ``−866025.4 eV``.
    An ``.lte`` carries **no species**, so it is a reader option (elegant's own
    default is the electron) and ``EQUIVALENT:SPECIES_ASSUMED`` is recorded
    whenever the deck has RF.

``RFCA`` defaults ``FREQ=500 MHz``, ``CHANGE_P0=0``
    an ``.lte`` with no ``FREQ`` really is a 500 MHz cavity in elegant, so the
    reader fills it in and records ``EQUIVALENT:RFCA_DEFAULT_FREQ``.

Statement syntax (measured, and *not* what every third-party parser assumes):
``!`` starts a comment; ``&`` at end of line continues; ``;`` is **not** a
statement separator (``q: quad, l=1; l1: line=(q)`` makes elegant complain about
a parameter named ``L1:LINE``); a bare trailing ``,`` is not a continuation for
elegant either — it silently drops the rest — but HELIX-era decks rely on it, so
this reader continues on a trailing comma only when the next code line does not
start a new statement, which is strictly closer to the author's intent and never
merges two definitions.  Beam lines may nest inline sub-lists (``2*(Q,D)``),
which are kept as anonymous IR lines rather than flattened.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.formats.elegant.naming import RESERVED, parse_tags
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    Octupole,
    Provenance,
    Quadrupole,
    RFCavity,
    Sextupole,
    Solenoid,
    SolenoidP,
    Taylor,
)
from lattix.ir.expr import Expression, ExpressionError, evaluate, evaluate_rpn
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.reference import ReferenceParticle, Species
from lattix.ir.reference import species as get_species
from lattix.ir.reference_tag import parse_reference_tag
from lattix.ir.rf import phase_from_elegant_deg
from lattix.ir.walk import propagate

#: elegant's own ``RFCA``/``RFCW`` frequency default (measured from its parameter table).
DEFAULT_RF_FREQUENCY_HZ = 500e6
#: elegant's ``SBEN``/``CSBEND`` ``FINT`` default (measured; the PLAN's 0.45 is wrong).
DEFAULT_FINT = 0.5
#: elegant decks carry no beam; these are the reader's stated assumptions.
DEFAULT_SPECIES = "electron"
DEFAULT_KINETIC_ENERGY_eV = 1e9

# --------------------------------------------------------------------------- syntax
_STATEMENT_START = re.compile(
    r"""^\s*(?: % | use\b | return\b | \#include\b
              | [^\s,=]+ \s* \[            # name[prop] = value
              | [^\s,=]+ \s* : )""",
    re.IGNORECASE | re.VERBOSE,
)
#: an unmistakable new definition: ``NAME: <known elegant type or LINE>``, or a
#: statement keyword.  Used only to stop an *unterminated* beam line from swallowing the
#: rest of the deck when its continuation line turns out to be entirely commented out
#: (the PIP-II BTL/BAL anchors do exactly that).  Deliberately stricter than
#: ``_STATEMENT_START`` so a template definition (``Q1: QBASE, …``) is never mistaken
#: for one while a genuine attribute list is being joined.
_NEW_DEFINITION = re.compile(
    r"""^\s*(?: % | use\b | return\b | \#include\b
               | (?:"[^"]*"|[^\s:,=]+) \s* : \s* (?:""" + "|".join(sorted(RESERVED, key=len,
                                                                          reverse=True)) + r""")\b )""",
    re.IGNORECASE | re.VERBOSE,
)
_USE_RE = re.compile(r"^\s*use\s*[,\s]\s*(.+?)\s*$", re.IGNORECASE)
_LINE_RE = re.compile(r"^\s*(?P<name>\"[^\"]*\"|[^\s:,]+)\s*:\s*line\s*=\s*(?P<body>\(.*\))\s*$",
                      re.IGNORECASE)
_LINE_OPEN_RE = re.compile(r"^\s*(?:\"[^\"]*\"|[^\s:,]+)\s*:\s*line\s*=\s*\(", re.IGNORECASE)
_ELEMENT_RE = re.compile(r"^\s*(?P<name>\"[^\"]*\"|[^\s:,]+)\s*:\s*(?P<type>[A-Za-z][\w]*)"
                         r"\s*(?:,(?P<body>.*))?$", re.IGNORECASE)
_OVERRIDE_RE = re.compile(r"^\s*(?P<name>\"[^\"]*\"|[^\s\[]+)\s*\[\s*(?P<prop>[\w.]+)\s*\]"
                          r"\s*=\s*(?P<value>.+?)\s*$")
_VAR_RE = re.compile(r"^\s*(?P<name>[A-Za-z][\w.]*)\s*=\s*(?P<value>.+?)\s*$")
_REPEAT_RE = re.compile(r"^(\d+)\s*\*\s*(.*)$")
_RETURN_RE = re.compile(r"^\s*return\b", re.IGNORECASE)


def logical_statements(text: str) -> list[tuple[int, str]]:
    """``(1-based line number, joined statement)`` for every logical statement.

    ``!`` comments are stripped; ``&`` continues; an unbalanced ``(`` continues;
    a trailing ``,`` continues only when the next code line is not itself a new
    statement (see the module docstring).
    """
    codes: list[tuple[int, str]] = []
    for n, raw in enumerate(text.splitlines(), 1):
        code = raw.split("!", 1)[0]
        stripped = code.lstrip()
        if stripped.startswith("#") and not stripped.lower().startswith("#include"):
            code = ""
        codes.append((n, code.rstrip()))

    out: list[tuple[int, str]] = []
    buf = ""
    first = 0
    for i, (n, code) in enumerate(codes):
        if not code.strip():
            continue
        piece = code.rstrip()
        cont = piece.endswith("&")
        if cont:
            piece = piece[:-1].rstrip()
        if buf and buf.count("(") > buf.count(")") and _NEW_DEFINITION.match(piece):
            out.append((first, buf))          # unterminated beam line: do not swallow the deck
            buf = ""
        if not buf:
            first = n
        buf = f"{buf} {piece.strip()}".strip() if buf else piece.strip()
        if not cont and not _next_starts_statement(codes, i + 1):
            # an attribute list or a beam line split over lines without a trailing '&'
            cont = buf.count("(") > buf.count(")") or buf.endswith(",")
        if not cont and buf:
            out.append((first, buf))
            buf = ""
    if buf:
        out.append((first, buf))
    return out


def _next_starts_statement(codes: list[tuple[int, str]], j: int) -> bool:
    for _, code in codes[j:]:
        if not code.strip():
            continue
        return bool(_STATEMENT_START.match(code))
    return True


def unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


def split_top(body: str, sep: str = ",") -> list[str]:
    """Split on *sep* at parenthesis depth 0, honouring double quotes."""
    out: list[str] = []
    depth = 0
    quoted = False
    cur: list[str] = []
    for ch in body:
        if ch == '"':
            quoted = not quoted
        if not quoted:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == sep and depth == 0:
                out.append("".join(cur))
                cur = []
                continue
        cur.append(ch)
    out.append("".join(cur))
    return out


def split_attrs(body: str | None) -> tuple[dict[str, str], list[str]]:
    """``({KEY: raw text}, [value-less tokens])`` from an element's attribute list."""
    attrs: dict[str, str] = {}
    bare: list[str] = []
    for piece in split_top(body or ""):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            bare.append(piece)
            continue
        key, val = piece.split("=", 1)
        attrs[key.strip().upper()] = val.strip()
    return attrs, bare


def evaluate_value(text: str, variables: dict[str, float]) -> float | None:
    """A number, an RPN expression or an infix expression — ``None`` if unresolvable."""
    s = unquote(str(text)).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    for fn in (evaluate_rpn, evaluate):
        try:
            return float(fn(s, variables))
        except (ExpressionError, ValueError, ZeroDivisionError, TypeError, KeyError, OverflowError):
            continue
    return None


# --------------------------------------------------------------------------- model
@dataclass
class _Def:
    """One ``name: TYPE, attrs`` definition, attributes still as source text."""

    name: str
    type: str
    attrs: dict[str, str] = field(default_factory=dict)
    line: int = 0
    bare: list[str] = field(default_factory=list)


class _AttrView:
    """Evaluate a definition's attributes, remembering which ones were used."""

    def __init__(self, defn: _Def, variables: dict[str, float]):
        self.defn = defn
        self.variables = variables
        self.consumed: set[str] = set()
        self.unresolved: list[str] = []

    def has(self, key: str) -> bool:
        return key.upper() in self.defn.attrs

    def take(self, *keys: str) -> None:
        for k in keys:
            self.consumed.add(k.upper())

    def num(self, key: str, default: float = 0.0) -> float:
        k = key.upper()
        self.consumed.add(k)
        text = self.defn.attrs.get(k)
        if text is None:
            return default
        v = evaluate_value(text, self.variables)
        if v is None:
            self.unresolved.append(f"{key}={text}")
            return default
        return v

    def flag(self, key: str, default: int = 0) -> int:
        return int(round(self.num(key, float(default))))

    def text(self, key: str) -> str | None:
        k = key.upper()
        self.consumed.add(k)
        t = self.defn.attrs.get(k)
        return None if t is None else unquote(t)


# --------------------------------------------------------------------------- mapping
#: elegant type keyword -> builder method suffix on :class:`Reader`.
TYPE_MAP: dict[str, str] = {
    "DRIF": "drift", "DRIFT": "drift", "EDRIFT": "drift",
    "CSRDRIFT": "collective_drift", "CSRDRIF": "collective_drift",
    "LSCDRIFT": "collective_drift", "LSCDRIF": "collective_drift",
    "QUAD": "quadrupole", "QUADRUPOLE": "quadrupole", "KQUAD": "quadrupole",
    "SEXT": "sextupole", "SEXTUPOLE": "sextupole", "KSEXT": "sextupole",
    "OCTU": "octupole", "OCTUPOLE": "octupole", "KOCT": "octupole",
    "MULT": "multipole", "MULTIPOLE": "multipole",
    "SBEN": "bend", "SBEND": "bend", "CSBEND": "bend", "CSRCSBEND": "bend",
    "CSRCSBEN": "bend", "NIBEND": "bend", "CCBEND": "bend",
    "RBEN": "rbend", "RBEND": "rbend",
    "SOLE": "solenoid", "SOLENOID": "solenoid",
    "RFCA": "rfcavity", "RFCW": "rfcavity",
    "HKICK": "hkick", "HKIC": "hkick", "EHKICK": "hkick",
    "VKICK": "vkick", "VKIC": "vkick", "EVKICK": "vkick",
    "KICK": "kicker", "KICKER": "kicker", "EKICKER": "kicker",
    "ECOL": "collimator", "RCOL": "collimator", "MAXAMP": "maxamp",
    "MARK": "marker", "MARKER": "marker",
    "MONI": "instrument", "MONITOR": "instrument", "HMON": "instrument",
    "VMON": "instrument", "WATCH": "instrument",
    "EMATRIX": "ematrix",
}

#: types with no clean IR equivalent -> ``Marker`` + ``DROPPED:UNSUPPORTED_ELEGANT_TYPE``.
UNSUPPORTED_TYPES: frozenset[str] = frozenset({
    "RFDF", "TWLA", "TWMTA", "TMCF", "MODRF", "RAMPRF", "RFTMEZ0", "SHRFDF",
    "MRFDF", "RFTM110", "CEPL", "TWPL", "RMDF", "BUMPER", "MBUMPER",
})

#: beam-data elements -> ``Directive(role="beam")`` + ``LOSSY:BEAM_DATA_DROPPED``.
BEAM_DATA_TYPES: frozenset[str] = frozenset({
    "CHARGE", "WAKE", "TRWAKE", "ZLONGIT", "ZTRANSVERSE", "RFMODE", "TRFMODE",
    "FRFMODE", "FTRFMODE", "LRWAKE", "CORGPIPE", "SCMULT", "RIMULT",
})

#: physics / bookkeeping elements -> ``Marker`` + ``LOSSY:PHYSICS_ELEMENT_DROPPED``.
PHYSICS_TYPES: frozenset[str] = frozenset({
    "SCATTER", "DSCATTER", "TSCATTER", "SREFFECTS", "IBSCATTER", "MALIGN",
    "ROTATE", "TWISSELEMENT", "ILMATRIX", "ENERGY", "CENTER", "RECIRC",
    "SCRIPT", "PFILTER", "REMCOR", "MAGNIFY", "REFLECT", "CLEAN", "EMITTANCE",
    "MATTER", "PEPPOT", "STRAY", "SAMPLE", "TRCOUNT", "IONEFFECTS", "FLOOR",
})

#: ``Instrument.family`` per diagnostic type keyword.
INSTRUMENT_FAMILY: dict[str, str] = {
    "MONI": "BPM", "MONITOR": "BPM", "HMON": "HMON", "VMON": "VMON", "WATCH": "WATCH",
}


def _num(x: float) -> str:
    s = f"{float(x):.15g}"
    return "0" if s in ("-0", "-0.0") else s


class Reader:
    """``Reader().read(path)`` -> ``(Lattice, FidelityReport)``."""

    format = "elegant"

    def __init__(self) -> None:
        #: charge of the reference species; sets the RFCA crest sign.
        self.charge = -1

    # ------------------------------------------------------------------
    def read(self, path: Path, *, line: str | None = None, strict: bool = False,
             species: str | Species | None = None, kinetic_energy_eV: float | None = None,
             frequency_Hz: float | None = None) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        rep = FidelityReport(source_format="elegant", source_file=str(path))

        variables: dict[str, Variable] = {}
        values: dict[str, float] = {}
        defs: dict[str, _Def] = {}          # UPPER name -> definition
        order: list[str] = []               # definition order (UPPER names)
        lines: dict[str, Line] = {}         # UPPER name -> line
        line_order: list[str] = []
        use_root: str | None = None
        overrides = 0

        for lineno, stmt in logical_statements(text):
            head = stmt.lstrip()
            if head.startswith("%"):
                self._store(head[1:], variables, values, rep, lineno)
                continue
            if head.lower().startswith("#include"):
                rep.dropped("ELEGANT_INCLUDE", f"{stmt!r} is not followed (lattix reads one file)",
                            element=None, kind=None, line=lineno)
                continue
            if _RETURN_RE.match(head):
                break
            m = _USE_RE.match(head)
            if m:
                use_root = unquote(m.group(1).rstrip(","))
                continue
            if _LINE_OPEN_RE.match(stmt) and stmt.count("(") > stmt.count(")"):
                missing = stmt.count("(") - stmt.count(")")
                rep.lossy("UNTERMINATED_LINE",
                          f"beam line statement is missing {missing} ')' (a continuation line was "
                          "entirely commented out); closed at the last member read",
                          element=None, kind=None, line=lineno)
                stmt = stmt.rstrip().rstrip(",") + ")" * missing
            m = _LINE_RE.match(stmt)
            if m:
                name = unquote(m.group("name"))
                items = self._line_items(m.group("body")[1:-1], name, lines, line_order)
                if name.upper() in lines:
                    rep.equivalent("LINE_REDEFINED",
                                   f"beam line {name!r} is defined twice; the last one wins",
                                   element=name, kind=None, line=lineno)
                    line_order.remove(name.upper())
                lines[name.upper()] = Line(name=name, items=items)
                line_order.append(name.upper())
                continue
            m = _OVERRIDE_RE.match(stmt)
            if m:
                key = unquote(m.group("name")).upper()
                if key in defs:
                    defs[key].attrs[m.group("prop").upper()] = m.group("value").strip()
                    overrides += 1
                else:
                    rep.dropped("OVERRIDE_UNKNOWN_ELEMENT",
                                f"{stmt!r} sets a property of an element that is not defined",
                                element=unquote(m.group("name")), kind=None, line=lineno)
                continue
            m = _ELEMENT_RE.match(stmt)
            if m:
                self._define(m, defs, order, rep, lineno)
                continue
            m = _VAR_RE.match(stmt)
            if m:
                v = evaluate_value(m.group("value"), values)
                if v is not None:
                    nm = m.group("name")
                    values[nm] = values[nm.lower()] = values[nm.upper()] = v
                    variables[nm] = Variable(value=v, expression=Expression(
                        text=m.group("value").strip(), dialect="infix"))
                    continue
            rep.dropped("UNPARSED_STATEMENT", f"statement not understood: {stmt!r}",
                        element=None, kind=None, line=lineno)

        if overrides:
            rep.equivalent("ELEMENT_PROPERTY_OVERRIDE",
                           f"{overrides} 'name[prop]=value' override(s) folded into the element "
                           "definitions (real elegant ignores them inside a .lte)",
                           element=None, kind=None, count=overrides)

        # -- reference particle --------------------------------------------
        tag = parse_reference_tag(text)
        tagged = species_tagged = False
        if species is None and tag is not None:
            species, tagged, species_tagged = tag.species, True, True
        if kinetic_energy_eV is None and tag is not None:
            kinetic_energy_eV, tagged = tag.kinetic_energy_eV, True
        if frequency_Hz is None and tag is not None and tag.rf_frequency_Hz:
            frequency_Hz = tag.rf_frequency_Hz
        sp = get_species(species if species is not None else DEFAULT_SPECIES)
        self.charge = sp.charge
        ke = DEFAULT_KINETIC_ENERGY_eV if kinetic_energy_eV is None else float(kinetic_energy_eV)
        ref = ReferenceParticle(species=sp, kinetic_energy_eV=ke, rf_frequency_Hz=frequency_Hz)
        if tagged:
            rep.equivalent("REFERENCE_FROM_TAG", f"reference particle ({sp.name}, {_num(ke)} eV kinetic) taken "
                           "from the deck's '! lattix: reference' tag", element=None, kind=None)
        if not species_tagged and any(d.type in ("RFCA", "RFCW") or d.type in UNSUPPORTED_TYPES for d in defs.values()):
            rep.equivalent("SPECIES_ASSUMED",
                           f"an .lte carries no beam: RF phases were read as {sp.name!r} "
                           f"(crest {'+90' if sp.charge < 0 else '-90'} deg); pass species= to "
                           "change it", element=None, kind=None, species=sp.name, charge=sp.charge)
        if kinetic_energy_eV is None:
            rep.equivalent("ENERGY_ASSUMED",
                           "an .lte carries no beam energy (it lives in the .ele run file); "
                           f"normalized strengths were converted at {_num(ke)} eV kinetic",
                           element=None, kind=None, kinetic_energy_eV=ke, species=sp.name)

        # -- elements -------------------------------------------------------
        tags = parse_tags(text)
        lat = Lattice(name=path.stem or "elegant", reference=ref, variables=variables)
        lat.meta["source_format"] = "elegant"
        pending: list[tuple[Element, str, int, float]] = []
        for key in order:
            defn = defs[key]
            el = self._build(defn, values, str(path), tags, rep, pending)
            lat.elements[el.name] = el
        by_upper = {n.upper(): n for n in lat.elements}

        # -- lines ----------------------------------------------------------
        for key in line_order:
            src = lines[key]
            lat.lines[src.name] = Line(name=src.name, items=[
                LineItem(ref=by_upper.get(it.ref.upper(), it.ref), repeat=it.repeat,
                         reverse=it.reverse) for it in src.items])
        lat.use = self._root(line, use_root, line_order, lat, rep)
        self._check_line_refs(lat, rep)

        # -- rigidity: normalized strengths need Bρ at each element's entrance --
        self._apply_rigidity(lat, pending, rep)
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _store(body: str, variables: dict[str, Variable], values: dict[str, float],
               rep: FidelityReport, lineno: int) -> None:
        """``% <rpn> sto <NAME>`` — elegant's RPN variable store."""
        toks = body.split()
        low = [t.lower() for t in toks]
        if "sto" not in low:
            rep.dropped("RPN_COMMAND_DROPPED",
                        f"'% {body.strip()}' is an rpn command without 'sto' and has no IR meaning",
                        element=None, kind=None, line=lineno)
            return
        i = low.index("sto")
        if i + 1 >= len(toks):
            rep.dropped("RPN_STORE_MALFORMED",
                        f"'% {body.strip()}' has no variable name after 'sto'",
                        element=None, kind=None, line=lineno)
            return
        name = toks[i + 1]
        expr = " ".join(toks[:i])
        v = evaluate_value(expr, values)
        if v is None:
            rep.dropped("RPN_STORE_UNRESOLVED", f"cannot evaluate '% {body.strip()}'",
                        element=None, kind=None, line=lineno)
            return
        values[name] = values[name.lower()] = values[name.upper()] = v
        variables[name] = Variable(value=v, expression=Expression(text=expr, dialect="rpn"))

    @staticmethod
    def _define(m: re.Match, defs: dict[str, _Def], order: list[str], rep: FidelityReport,
                lineno: int) -> None:
        name = unquote(m.group("name"))
        etype = m.group("type").upper()
        attrs, bare = split_attrs(m.group("body"))
        base = defs.get(etype)
        if base is not None:                    # template: the type token names a defined element
            merged = dict(base.attrs)
            merged.update(attrs)
            attrs = merged
            etype = base.type
        key = name.upper()
        if key in defs:
            rep.equivalent("ELEMENT_REDEFINED",
                           f"element {name!r} is defined twice; the last definition wins",
                           element=name, kind=None, line=lineno)
            order.remove(key)
        defs[key] = _Def(name=name, type=etype, attrs=attrs, line=lineno, bare=bare)
        order.append(key)

    def _line_items(self, body: str, parent: str, lines: dict[str, Line],
                    line_order: list[str]) -> list[LineItem]:
        items: list[LineItem] = []
        for tok in split_top(body):
            tok = tok.strip()
            if not tok:
                continue
            reverse = False
            repeat = 1
            while True:
                if tok.startswith("-"):
                    reverse = not reverse
                    tok = tok[1:].lstrip()
                    continue
                rm = _REPEAT_RE.match(tok)
                if rm:
                    repeat *= int(rm.group(1))
                    tok = rm.group(2).lstrip()
                    continue
                break
            if tok.startswith("(") and tok.endswith(")"):
                sub = f"{parent}__{len(line_order) + 1}"
                while sub.upper() in lines:
                    sub += "_"
                sub_items = self._line_items(tok[1:-1], sub, lines, line_order)
                lines[sub.upper()] = Line(name=sub, items=sub_items)
                line_order.append(sub.upper())
                tok = sub
            items.append(LineItem(ref=unquote(tok), repeat=repeat, reverse=reverse))
        return items

    @staticmethod
    def _root(requested: str | None, use_root: str | None, line_order: list[str],
              lat: Lattice, rep: FidelityReport) -> str | None:
        by_upper = {n.upper(): n for n in lat.lines}
        for candidate, why in ((requested, "line="), (use_root, "USE")):
            if candidate:
                nm = by_upper.get(candidate.upper())
                if nm:
                    return nm
                rep.dropped("UNKNOWN_ROOT_LINE",
                            f"{why} names {candidate!r}, which is not a beam line in this deck",
                            element=candidate, kind=None)
        for key in reversed(line_order):
            nm = by_upper.get(key)
            if nm:
                rep.equivalent("ROOT_LINE_IS_LAST",
                               f"no USE statement; elegant uses the last beam line ({nm!r})",
                               element=nm, kind=None)
                return nm
        rep.dropped("NO_BEAMLINE", "the deck defines no 'line=(…)'; the lattice has no root line",
                    element=None, kind=None)
        return None

    @staticmethod
    def _check_line_refs(lat: Lattice, rep: FidelityReport) -> None:
        """Drop members no definition backs, so :meth:`Lattice.flatten` stays usable."""
        known = {n.upper() for n in lat.elements} | {n.upper() for n in lat.lines}
        for ln in lat.lines.values():
            kept = []
            for it in ln.items:
                if it.ref.upper() in known:
                    kept.append(it)
                else:
                    rep.dropped("UNDEFINED_LINE_MEMBER",
                                f"beam line {ln.name!r} references {it.ref!r}, which the deck never "
                                "defines; the reference was dropped",
                                element=it.ref, kind=None, line_name=ln.name)
            ln.items = kept

    # ------------------------------------------------------------------ rigidity
    @staticmethod
    def _apply_rigidity(lat: Lattice, pending: list[tuple[Element, str, int, float]],
                        rep: FidelityReport) -> None:
        """Turn normalized strengths into lab fields with Bρ at each element's entrance."""
        if not pending:
            return
        brho: dict[int, float] = {}
        if lat.use:
            try:
                for p in propagate(lat):
                    ref = p.ref_in or lat.reference
                    if id(p.element) not in brho:
                        brho[id(p.element)] = ref.brho_signed
                    elif abs(brho[id(p.element)] - ref.brho_signed) > 1e-12 * abs(ref.brho_signed):
                        rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                                       "one definition is used at two reference energies; the "
                                       "first occurrence's rigidity converts its normalized "
                                       "strengths", element=p.element.name, kind=p.element.kind)
            except (KeyError, ValueError) as e:
                rep.equivalent("WALK_FAILED",
                               f"could not propagate the reference particle ({e}); normalized "
                               "strengths use the rigidity at the lattice start",
                               element=None, kind=None)
        start = lat.reference.brho_signed
        for el, target, order, value in pending:
            b = brho.get(id(el), start)
            if target == "Bn":
                el.multipole.Bn[order] = value * b
            elif target == "BnL":
                el.multipole.BnL[order] = value * b
            elif target == "Bs":
                el.multipole.Bs[order] = value * b
            else:
                el.solenoid.Bsol_T = value * b

    # ------------------------------------------------------------------ elements
    def _build(self, defn: _Def, values: dict[str, float], file: str, tags: dict,
               rep: FidelityReport, pending: list) -> Element:
        a = _AttrView(defn, values)
        handler = TYPE_MAP.get(defn.type)
        if handler is not None:
            el = getattr(self, f"_el_{handler}")(defn, a, rep, pending)
        elif defn.type in BEAM_DATA_TYPES:
            a.take(*defn.attrs)
            el = Directive(name=defn.name, format="elegant", card=defn.type, role="beam",
                           args=[f"{k}={v}" for k, v in defn.attrs.items()])
            rep.lossy("BEAM_DATA_DROPPED",
                      f"{defn.type} carries beam/impedance data, not geometry; kept as an elegant "
                      "directive and dropped by every other writer",
                      element=defn.name, kind="Directive", line=defn.line, elegant_type=defn.type)
        elif defn.type in PHYSICS_TYPES:
            el = Marker(name=defn.name)
            rep.lossy("PHYSICS_ELEMENT_DROPPED",
                      f"{defn.type} models a physics process, not optics; written as a marker",
                      element=defn.name, kind="Marker", line=defn.line, elegant_type=defn.type)
        elif defn.type in UNSUPPORTED_TYPES:
            el = Marker(name=defn.name)
            rep.dropped("UNSUPPORTED_ELEGANT_TYPE",
                        f"{defn.type} has no IR equivalent; written as a marker",
                        element=defn.name, kind="Marker", line=defn.line, elegant_type=defn.type)
        else:
            length = a.num("L")
            el = Drift(name=defn.name, length=length) if length else Marker(name=defn.name)
            rep.dropped("UNSUPPORTED_ELEGANT_TYPE",
                        f"unknown elegant type {defn.type!r}; written as a "
                        f"{'drift' if length else 'marker'}",
                        element=defn.name, kind=el.kind, line=defn.line, elegant_type=defn.type)

        self._finish(el, defn, a, file, tags, rep)
        return el

    @staticmethod
    def _finish(el: Element, defn: _Def, a: _AttrView, file: str, tags: dict,
                rep: FidelityReport) -> None:
        original = tags.get(defn.name.upper(), {})
        el.provenance = Provenance(format="elegant", file=file, line=defn.line,
                                   original_name=original.get("name", defn.name),
                                   original_type=(original.get("type") if original else defn.type))
        el.native["elegant"] = {"type": defn.type, "attrs": dict(defn.attrs),
                                "consumed": sorted(a.consumed & set(defn.attrs))}
        unknown = sorted(set(defn.attrs) - a.consumed)
        if unknown:
            rep.equivalent("UNMAPPED_ELEGANT_ATTRIBUTE",
                           f"{defn.type} attribute(s) {', '.join(unknown)} have no IR field; kept "
                           "in native['elegant'] and re-emitted by the elegant writer only",
                           element=defn.name, kind=el.kind, line=defn.line, attrs=unknown)
        if defn.bare:
            rep.lossy("BARE_ATTRIBUTE_TOKEN",
                      f"{defn.type} attribute list has value-less token(s) "
                      f"{', '.join(defn.bare)} (elegant itself refuses such a deck)",
                      element=defn.name, kind=el.kind, line=defn.line, tokens=defn.bare)
        if a.unresolved:
            rep.lossy("UNRESOLVED_EXPRESSION",
                      f"could not evaluate {', '.join(a.unresolved)}; the elegant default was used",
                      element=defn.name, kind=el.kind, line=defn.line, attrs=a.unresolved)

    # -- per-kind builders --------------------------------------------------
    @staticmethod
    def _shift(a: _AttrView, *, tilt: float = 0.0, x_rot: float = 0.0,
               y_rot: float = 0.0) -> BodyShiftP | None:
        s = BodyShiftP(x_offset=a.num("DX"), y_offset=a.num("DY"), z_offset=a.num("DZ"),
                       x_rot=x_rot, y_rot=y_rot, tilt=tilt)
        return None if s.is_zero() else s

    @staticmethod
    def _tilt(a: _AttrView, order: int) -> dict[int, float]:
        t = a.num("TILT")
        return {order: t} if t else {}

    def _el_drift(self, defn, a, rep, pending) -> Element:
        rep.exact(defn.name, "Drift")
        return Drift(name=defn.name, length=a.num("L"))

    def _el_collective_drift(self, defn, a, rep, pending) -> Element:
        rep.equivalent("COLLECTIVE_DRIFT_AS_DRIFT",
                       f"{defn.type} is a drift plus a collective-effect model; the geometry is "
                       "exact, the CSR/LSC model lives only in native['elegant']",
                       element=defn.name, kind="Drift", elegant_type=defn.type)
        return Drift(name=defn.name, length=a.num("L"))

    def _el_quadrupole(self, defn, a, rep, pending) -> Element:
        el = Quadrupole(name=defn.name, length=a.num("L"),
                        multipole=MagneticMultipoleP(tilt=self._tilt(a, 1)))
        pending.append((el, "Bn", 1, a.num("K1")))
        if a.has("K2"):                       # elegant lets a QUAD carry a sextupole term
            pending.append((el, "Bn", 2, a.num("K2")))
        el.shift = self._shift(a, x_rot=-a.num("PITCH"), y_rot=a.num("YAW"))
        rep.exact(defn.name, "Quadrupole")
        return el

    def _el_sextupole(self, defn, a, rep, pending) -> Element:
        el = Sextupole(name=defn.name, length=a.num("L"),
                       multipole=MagneticMultipoleP(tilt=self._tilt(a, 2)))
        pending.append((el, "Bn", 2, a.num("K2")))
        if a.has("K1"):
            pending.append((el, "Bn", 1, a.num("K1")))
        if a.has("J1"):
            pending.append((el, "Bs", 1, a.num("J1")))
        el.shift = self._shift(a, x_rot=-a.num("PITCH"), y_rot=a.num("YAW"))
        rep.exact(defn.name, "Sextupole")
        return el

    def _el_octupole(self, defn, a, rep, pending) -> Element:
        el = Octupole(name=defn.name, length=a.num("L"),
                      multipole=MagneticMultipoleP(tilt=self._tilt(a, 3)))
        pending.append((el, "Bn", 3, a.num("K3")))
        el.shift = self._shift(a, x_rot=-a.num("PITCH"), y_rot=a.num("YAW"))
        rep.exact(defn.name, "Octupole")
        return el

    def _el_multipole(self, defn, a, rep, pending) -> Element:
        order = a.flag("ORDER", 1)
        knl = a.num("KNL") * a.num("FACTOR", 1.0)
        el = Multipole(name=defn.name, length=a.num("L"),
                       multipole=MagneticMultipoleP(tilt=self._tilt(a, order)))
        pending.append((el, "BnL", order, knl))
        el.shift = self._shift(a)
        if el.length:
            rep.equivalent("THICK_MULT_AS_THIN",
                           "elegant MULT has a length but delivers an integrated kick; the IR "
                           "keeps both the length and the integrated strength",
                           element=defn.name, kind="Multipole", length=el.length)
        else:
            rep.exact(defn.name, "Multipole")
        return el

    def _bend(self, defn, a, rep, pending, *, rect: bool) -> Bend:
        angle = a.num("ANGLE")
        length = a.num("L")
        e1, e2 = a.num("E1"), a.num("E2")
        if rect and angle:
            # measured: elegant rewrites RBEN as SBEN with L_arc = L_chord·(θ/2)/sin(θ/2)
            # and E1/E2 += θ/2 (identical to MAD-X rbarc).
            half = angle / 2.0
            length = length * half / math.sin(half)
            e1 += half
            e2 += half
        fint = a.num("FINT", DEFAULT_FINT)
        fint1, fint2 = a.num("FINT1", -1.0), a.num("FINT2", -1.0)
        el = Bend(name=defn.name, length=length,
                  bend=BendP(angle=angle, e1=e1, e2=e2, hgap=a.num("HGAP"),
                             tilt_ref=a.num("TILT"), rect=rect,
                             edge_int1=fint if fint1 < 0 else fint1,
                             edge_int2=None if fint2 < 0 else fint2))
        for order, key in ((1, "K1"), (2, "K2"), (3, "K3")):
            if a.has(key):
                pending.append((el, "Bn", order, a.num(key)))
            else:
                a.take(key)
        el.shift = self._shift(a, tilt=a.num("ETILT") * a.flag("ETILT_SIGN", 1),
                               x_rot=a.num("EPITCH"), y_rot=a.num("EYAW"))
        if a.num("H1") or a.num("H2"):
            rep.lossy("BEND_POLE_CURVATURE_DROPPED",
                      "H1/H2 pole-face curvature has no IR field; kept in native['elegant']",
                      element=defn.name, kind="Bend", line=defn.line)
        else:
            rep.exact(defn.name, "Bend")
        return el

    def _el_bend(self, defn, a, rep, pending) -> Element:
        return self._bend(defn, a, rep, pending, rect=False)

    def _el_rbend(self, defn, a, rep, pending) -> Element:
        el = self._bend(defn, a, rep, pending, rect=True)
        rep.equivalent("RBEN_CHORD_TO_ARC",
                       "elegant RBEN 'L' is the chord: the IR stores the arc "
                       "L*(angle/2)/sin(angle/2) with angle/2 added to both pole faces "
                       "(measured against elegant's own &save_lattice)",
                       element=defn.name, kind="Bend", arc_length=el.length)
        return el

    def _el_solenoid(self, defn, a, rep, pending) -> Element:
        el = Solenoid(name=defn.name, length=a.num("L"), solenoid=SolenoidP())
        if a.has("B"):
            el.solenoid.Bsol_T = a.num("B")
            a.take("KS")
        else:
            pending.append((el, "Bsol", 0, a.num("KS")))
        el.shift = self._shift(a)
        rep.exact(defn.name, "Solenoid")
        return el

    def _el_rfcavity(self, defn, a, rep, pending) -> Element:
        freq = a.num("FREQ", DEFAULT_RF_FREQUENCY_HZ)
        if not a.has("FREQ"):
            rep.equivalent("RFCA_DEFAULT_FREQ",
                           f"{defn.type} has no FREQ; elegant's own default "
                           f"{_num(DEFAULT_RF_FREQUENCY_HZ)} Hz was filled in",
                           element=defn.name, kind="RFCavity", frequency_Hz=DEFAULT_RF_FREQUENCY_HZ)
        length = a.num("L")
        el = RFCavity(name=defn.name, length=length,
                      rf=RFP(voltage_V=a.num("VOLT"), frequency_Hz=freq,
                             phase_rad=phase_from_elegant_deg(a.num("PHASE"), self.charge),
                             L_active_m=length or None))
        el.shift = self._shift(a)
        if not a.flag("CHANGE_P0"):
            rep.equivalent("RFCA_NO_P0_CHANGE",
                           "CHANGE_P0=0: elegant does not move its reference momentum through this "
                           "cavity, while the IR walk applies V*cos(phase)",
                           element=defn.name, kind="RFCavity", voltage_V=el.rf.voltage_V)
        else:
            rep.exact(defn.name, "RFCavity")
        return el

    def _el_hkick(self, defn, a, rep, pending) -> Element:
        el = Kicker(name=defn.name, length=a.num("L"), hkick=a.num("KICK"),
                    electric=defn.type.startswith("E"))
        el.shift = self._shift(a)
        rep.exact(defn.name, "Kicker")
        self._kick_tilt(defn, a, rep)
        return el

    def _el_vkick(self, defn, a, rep, pending) -> Element:
        el = Kicker(name=defn.name, length=a.num("L"), vkick=a.num("KICK"),
                    electric=defn.type.startswith("E"))
        el.shift = self._shift(a)
        rep.exact(defn.name, "Kicker")
        self._kick_tilt(defn, a, rep)
        return el

    def _el_kicker(self, defn, a, rep, pending) -> Element:
        el = Kicker(name=defn.name, length=a.num("L"), hkick=a.num("HKICK"), vkick=a.num("VKICK"),
                    electric=defn.type.startswith("E"))
        el.shift = self._shift(a)
        rep.exact(defn.name, "Kicker")
        self._kick_tilt(defn, a, rep)
        return el

    @staticmethod
    def _kick_tilt(defn: _Def, a: _AttrView, rep: FidelityReport) -> None:
        tilt = a.num("TILT")
        if tilt:
            rep.lossy("KICKER_TILT_KEPT_NATIVE",
                      "a tilted corrector's kick plane is not rotated into hkick/vkick; TILT is "
                      "kept in native['elegant']",
                      element=defn.name, kind="Kicker", line=defn.line, tilt=tilt)

    def _el_collimator(self, defn, a, rep, pending) -> Element:
        shape = "ELLIPTICAL" if defn.type == "ECOL" else "RECTANGULAR"
        # elegant: X_MAX = 0 / Y_MAX = 0 (the defaults) mean "no limit in that plane"
        xm, ym = a.num("X_MAX"), a.num("Y_MAX")
        ap = ApertureP(shape=shape, x_limits=(-xm, xm) if xm else None,
                       y_limits=(-ym, ym) if ym else None) if (xm or ym) else None
        el = Collimator(name=defn.name, length=a.num("L"), aperture=ap)
        el.shift = self._shift(a)
        rep.exact(defn.name, "Collimator")
        return el

    def _el_maxamp(self, defn, a, rep, pending) -> Element:
        shape = "ELLIPTICAL" if a.flag("ELLIPTICAL") else "RECTANGULAR"
        el = Collimator(name=defn.name, length=0.0,
                        aperture=ApertureP(shape=shape,
                                           x_limits=(-a.num("X_MAX"), a.num("X_MAX")),
                                           y_limits=(-a.num("Y_MAX"), a.num("Y_MAX"))))
        rep.equivalent("MAXAMP_AS_COLLIMATOR",
                       "MAXAMP sets the aperture for everything downstream; the IR represents it "
                       "as a zero-length collimator at that point",
                       element=defn.name, kind="Collimator")
        return el

    def _el_marker(self, defn, a, rep, pending) -> Element:
        el = Marker(name=defn.name)
        el.shift = self._shift(a)
        rep.exact(defn.name, "Marker")
        return el

    def _el_instrument(self, defn, a, rep, pending) -> Element:
        el = Instrument(name=defn.name, length=a.num("L"),
                        family=INSTRUMENT_FAMILY.get(defn.type, "MONITOR"))
        el.shift = self._shift(a)
        if defn.type == "WATCH":
            el.params["filename"] = a.text("FILENAME") or ""
        rep.exact(defn.name, "Instrument")
        return el

    def _el_ematrix(self, defn, a, rep, pending) -> Element:
        order = a.flag("ORDER", 1)
        if order != 1:
            rep.lossy("EMATRIX_HIGHER_ORDER",
                      f"EMATRIX ORDER={order}: only the first-order map is representable; "
                      "written as a marker",
                      element=defn.name, kind="Marker", line=defn.line, order=order)
            return Marker(name=defn.name)
        matrix = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        for i in range(6):
            for j in range(6):
                key = f"R{i + 1}{j + 1}"
                if a.has(key):
                    matrix[i][j] = a.num(key)
                else:
                    a.take(key)
        offset = [a.num(f"C{i + 1}") for i in range(6)]
        el = Taylor(name=defn.name, length=a.num("L"), matrix=matrix, offset=offset)
        el.shift = self._shift(a, tilt=a.num("TILT"), x_rot=a.num("PITCH"), y_rot=a.num("YAW"))
        rep.exact(defn.name, "Taylor")
        return el
