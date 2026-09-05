"""MAD8 flat-file reader (PLAN §6 task 2.3).

Reads the MAD8 ``SAVELINE`` / hand-written flat dialect (``.lat`` / ``.FLAT``) that the
PIP-II BTL and BAL optics decks use.  Ported from HELIX ``linac_gen/io/mad8_parser.py``
(``_logical_lines``, ``_Mad8File``'s lazy ``:=`` resolver, ``_expand``, ``_root_line``,
``_build_mad8_element``, ``_declare_periods``) onto the lattix IR, with the differences
listed below — every one of them a fidelity *gain*, and every one recorded in the ledger.

Supported subset
----------------
* ``!`` comments and ``&`` end-of-line continuations (``_logical_lines``).
* ``name := expr`` / ``name = expr`` parameters resolved **lazily** with memoisation and
  cycle detection, including ``NAME[ATTR]`` element-attribute references.  Definitions may
  appear anywhere in the file (the PIP-II decks put them *after* the elements that use
  them).  The whole :mod:`lattix.ir.expr` function whitelist is available, so ``SQRT``,
  ``ABS``, ``SIN``, ``COS`` … work — the BAL deck's ``P0 := SQRT(E0*(2*MASS+E0))`` is
  exactly where HELIX's resolver stops (docs/corpus.md).
* ``name: TYPE, attr=expr, …`` element definitions, including MAD8 *class inheritance*
  (``QF2: QF, K1=…``: the parent's type and attributes are inherited).
* ``name: LINE=(A, B, -C, 2*D, (E, F))`` with nesting, reflection, repetition and
  anonymous sub-lines.  The root is the ``USE`` target if the file has one, else the
  unreferenced LINE with the largest expansion (``_root_line``, recorded as
  ``EQUIVALENT:AMBIGUOUS_ROOT_LINE``).

Conventions
-----------
* **Rigidity** — ``Bρ_signed = sign(q)·pc/(|q|c)``; every MAD normalized strength (K1, KS,
  KnL, kicks) is turned into the IR's lab field with it, so an H⁻ deck's gradients flip
  against a proton deck's (HELIX ``madx_parser._signed_brho``; external anchor: the legacy
  PIP-II BTL conversion header ``variable mad2tw -4.8828922``).
* **Rigidity resolution order** (PLAN 2.3) — ``brho=`` argument → a ``BRHO := …`` file
  parameter → a ``BEAM`` statement → a hard :class:`ValueError`.  There is no silent
  default: a wrong Bρ silently mis-scales every magnet.  Taking the rigidity from ``BRHO``
  records ``EQUIVALENT:RIGIDITY_FROM_BRHO`` with the kinetic energy it implies, and a
  ``BEAM`` statement that disagrees with it by more than 1e-6 adds
  ``EQUIVALENT:RIGIDITY_CONFLICT``.
* **Species** — a ``BEAM`` statement's ``PARTICLE`` / ``MASS`` / ``CHARGE`` wins (matched
  against the known species by mass and charge, else kept as a custom
  :class:`~lattix.ir.reference.Species`); the ``species=`` argument is the fallback for the
  many decks that carry only ``BRHO :=`` (default ``h-``, as in the PIP-II lineage).
* **RBEND length** — MAD8 has no ``OPTION, RBARC``: its RBEND ``L`` is the **arc**, unlike
  MAD-X's chord.  Verified against the PIP-II BTL anchor (``LBA := 2.408`` with
  ``BAANG := 0.11455892`` gives ρ = 21.019751 m = L/θ, the ρ in the TraceWin export).
  Pole faces are referenced to the chord as in MAD-X, so the IR gets ``e1 += angle/2``
  and ``BendP.rect = True``.
* **Vertical bends** — MAD8 encodes them as ``TILT = ±π/2``; the IR keeps the signed angle
  and ``BendP.tilt_ref`` verbatim (EXACT).  HELIX had to fold the tilt into a ``hv`` flag
  with ρ > 0 because its ``Dipole`` has no reference tilt.
* **Kickers** — ``KICKER``/``HKICKER``/``VKICKER`` become one thick
  :class:`~lattix.ir.elements.Kicker` that keeps both the deflection *and* the body length
  (EXACT).  HELIX split them into a marker plus a body drift and dropped the kick.
* **Monitors** — plain ``MONITOR`` is a generic instrument (ion pump, collimator flag …),
  ``HMONITOR``/``VMONITOR`` are beam-position monitors: family ``MONITOR`` vs ``BPM``, with
  the MAD8 type kept in ``native["mad8"]["type"]`` so a round trip is exact.
* **Periodicity** — the LINE hierarchy declares the machine's cell structure, which a flat
  TraceWin file loses.  With ``auto_periods=True`` the reader identifies FODO-type cells
  (LINE-valued grandchildren of the root), groups consecutive cells with identical
  transport signatures and identical significant-element counts, and brackets them with
  ``Directive(role="period_start"/"period_end", card="LATTICE"/"LATTICE_END")`` — the exact
  places HELIX's ``_declare_periods`` puts them, so the TraceWin writer emits the same
  ``LATTICE n 0`` cards.  The count *n* is the number of TraceWin cards the cell would emit
  (a ``Bend`` with pole faces is EDGE+BEND+EDGE = 3), which is what TraceWin counts.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    ApertureP,
    Bend,
    BendP,
    Collimator,
    Directive,
    Drift,
    Element,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    Octupole,
    Patch,
    Provenance,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    Sextupole,
    Solenoid,
)
from lattix.ir.energy_mode import ENERGY_MODES, restore_energy_mode, undo_phase_slip
from lattix.ir.expr import Expression, ExpressionError, evaluate
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name
from lattix.ir.reference_tag import parse_reference_tag
from lattix.ir.rf import phase_from_madx_lag

#: MAD8 identifiers start with a letter and may carry ``'`` (``QX'`` = dQx/dδ).
_IDENT = r"[A-Za-z][A-Za-z0-9_.']*"
#: an element/class name in the ``name: TYPE`` position (no apostrophes).
_TYPE = r"[A-Za-z][A-Za-z0-9_.]*"

_RE_PARAM = re.compile(rf"^({_IDENT})\s*:?=\s*(.+)$")
_RE_ELEMENT_HEAD = re.compile(rf"^{_IDENT}\s*:\s*[A-Za-z]")
_RE_LINE = re.compile(rf"^({_IDENT})\s*:\s*LINE\s*=\s*\((.*)\)\s*$", re.IGNORECASE)
_RE_ELEMENT = re.compile(rf"^({_IDENT})\s*:\s*({_TYPE})\s*(?:,\s*(.*))?$")
_RE_REPEAT = re.compile(rf"^(\d+)\s*\*\s*(-?)\s*({_IDENT}|\(.*\))$", re.DOTALL)
_RE_TITLE = re.compile(r'^TITLE\s*[, ]\s*(?:"([^"]*)"|\'([^\']*)\'|(.*))$', re.IGNORECASE)
_RE_NUMBER = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")
#: ``! lattix: name="…" type="…"`` tags written above a renamed definition.
_RE_TAG_COMMENT = re.compile(r'!\s*lattix:\s*(.*?)\s*$')
_ENERGY_MODE = re.compile(r"^!\s*lattix:\s*energy_mode=(\w+)", re.M)


def _energy_mode_tag(text: str) -> str | None:
    m = _ENERGY_MODE.search(text)
    return m.group(1).lower() if m else None
_RE_TAG_KV = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_RE_DEFN_NAME = re.compile(rf"^({_IDENT})\s*:")

#: MAD8 element keywords this reader maps (everything else is an unsupported type).
ELEMENT_TYPES: frozenset[str] = frozenset({
    "drift", "sbend", "rbend", "quadrupole", "sextupole", "octupole", "multipole",
    "solenoid", "rfcavity", "elseparator", "kicker", "hkicker", "vkicker", "monitor",
    "hmonitor", "vmonitor", "instrument", "blmonitor", "wire", "slmonitor",
    "rcollimator", "ecollimator", "marker", "srot", "yrot", "beambeam", "lump",
    "matrix", "gkick", "arbitelm", "twcavity", "rfmultipole", "profile",
})

#: MAD8 command statements that carry no lattice content; kept in ``meta`` only.
IGNORED_COMMANDS: frozenset[str] = frozenset({
    "title", "use", "saveline", "save", "return", "stop", "exit", "quit", "option",
    "select", "print", "value", "show", "plot", "setplot", "help", "system", "assign",
    "twiss", "survey", "match", "endmatch", "cell", "vary", "weight", "constraint",
    "migrad", "lmdif", "simplex", "track", "endtrack", "run", "emit", "ibs", "dynap",
    "split", "seqedit", "endedit", "beta0", "line", "sixtrack", "eoption", "efield",
    "ealign", "set", "harmon", "envelope", "correct", "micado", "getdisp", "usemonitor",
    "usekick", "sigma", "start", "observe", "noecho", "echo", "delete", "resplot",
    "structure", "aperture", "archive", "beam",
})

#: MAD8 attributes whose value is a name/string, never a number.
_STRING_ATTRS = frozenset({"particle", "type", "apertype", "label", "file", "filename",
                           "class", "sequence", "range"})

#: the "natural" tilt a bare ``TILT`` flag means, per multipole order (MAD8 §element).
_NATURAL_TILT = {"quadrupole": math.pi / 4, "sextupole": math.pi / 6,
                 "octupole": math.pi / 8, "sbend": math.pi / 2, "rbend": math.pi / 2}

#: MAD8 monitor keyword → IR ``Instrument.family``.
_MONITOR_FAMILY = {"monitor": "MONITOR", "hmonitor": "BPM", "vmonitor": "BPM",
                   "instrument": "INSTRUMENT", "blmonitor": "BLM", "slmonitor": "SLM",
                   "wire": "WIRE", "profile": "PROFILE"}

#: MAD8 particle names → lattix species keys (MAD8 knows only these four).
_PARTICLE_ALIAS = {"anti-proton": "antiproton", "antiproton": "antiproton",
                   "proton": "proton", "electron": "electron", "positron": "positron",
                   "hminus": "h-", "h-": "h-", "h_minus": "h-", "deuteron": "deuteron"}

_TOL_REL = 1e-6


def strip_comment(raw: str) -> str:
    """Drop a MAD8 ``!`` comment, honouring double-quoted strings."""
    out: list[str] = []
    in_quote = False
    for ch in raw:
        if ch == '"':
            in_quote = not in_quote
        elif ch == "!" and not in_quote:
            break
        out.append(ch)
    return "".join(out).rstrip()


def logical_lines(text: str) -> list[tuple[int, str]]:
    """``(line number, statement)`` with ``!`` comments stripped and ``&`` joined."""
    out: list[tuple[int, str]] = []
    buf = ""
    first = 0
    for no, raw in enumerate(text.splitlines(), 1):
        raw = strip_comment(raw)
        if not raw.strip() and not buf:
            continue
        if not buf:
            first = no
        buf += raw
        if buf.rstrip().endswith("&"):
            buf = buf.rstrip()[:-1]
            continue
        if buf.strip():
            out.extend((first, part) for part in _split_statements(buf))
        buf = ""
    if buf.strip():
        out.extend((first, part) for part in _split_statements(buf))
    return out


def _split_statements(text: str) -> list[str]:
    """MAD8 has no statement terminator, but MAD-X-flavoured flat files (PyORBIT's) end statements
    with ``;`` — split on top-level semicolons and drop empty pieces."""
    parts: list[str] = []
    depth = 0
    buf = ""
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == ";" and depth == 0:
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return [x.strip() for x in parts if x.strip()]


def split_top_level(body: str) -> list[str]:
    """Split on commas outside ``()``/``{}`` and double quotes."""
    parts: list[str] = []
    depth = 0
    quote = False
    cur: list[str] = []
    for ch in body:
        if ch == '"':
            quote = not quote
        elif not quote and ch in "({":
            depth += 1
        elif not quote and ch in ")}":
            depth -= 1
        if ch == "," and depth == 0 and not quote:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_attributes(body: str) -> dict[str, str | bool]:
    """``attr=expr, flag`` → raw strings (evaluated lazily: eager evaluation would
    silently coerce an unresolved forward reference to 0)."""
    attrs: dict[str, str | bool] = {}
    for part in split_top_level(body):
        if not part:
            continue
        if "=" not in part:
            attrs[part.strip().lower()] = True
            continue
        key, _, val = part.partition("=")
        attrs[key.rstrip(":").strip().lower()] = val.strip()
    return attrs


def parse_tags(text: str) -> dict[str, dict[str, str]]:
    """``{MAD8_NAME: {"name": original, "type": original_type}}`` from a deck's text.

    The writer puts the tag on its own comment line *above* the definition, so a
    definition broken across ``&`` continuation cards keeps its tag; a trailing tag on
    the statement itself is honoured too.
    """
    out: dict[str, dict[str, str]] = {}
    pending: dict[str, str] | None = None
    buf = ""
    for raw in text.splitlines():
        m = _RE_TAG_COMMENT.search(raw)
        kv = dict(_RE_TAG_KV.findall(m.group(1))) if m else {}
        code = strip_comment(raw)
        if not code.strip():
            if kv:
                pending = kv
            continue
        buf += code
        if buf.rstrip().endswith("&"):
            buf = buf.rstrip()[:-1]
            if kv:
                pending = kv
            continue
        stmt, buf = buf.strip(), ""
        tag = kv or pending
        pending = None
        m2 = _RE_DEFN_NAME.match(stmt)
        if m2 and tag:
            out[m2.group(1).upper()] = tag
    return out


def _looks_numeric(text: str) -> bool:
    return bool(_RE_NUMBER.match(text.strip()))


# ---------------------------------------------------------------------------
# Statement store
# ---------------------------------------------------------------------------
class _ElementDef:
    __slots__ = ("name", "type", "attrs", "line")

    def __init__(self, name: str, etype: str, attrs: dict[str, str | bool], line: int) -> None:
        self.name = name
        self.type = etype
        self.attrs = attrs
        self.line = line


class _Deck:
    """Classified statements of one MAD8 file (nothing is evaluated here)."""

    def __init__(self, text: str, rep: FidelityReport, warnings: list[str]) -> None:
        self.params: dict[str, str] = {}                 # UPPER name -> raw expression
        self.param_line: dict[str, int] = {}
        self.unparseable: dict[str, str] = {}            # names with ' (never evaluable)
        self.elems: dict[str, _ElementDef] = {}
        self.lines: dict[str, list[str]] = {}
        self.line_line: dict[str, int] = {}
        self.beam: dict[str, str | bool] = {}
        self.title: str | None = None
        self.use: str | None = None
        self.commands: list[str] = []
        self._anon = 0
        self.rep = rep
        self.warnings = warnings
        self._parse(text)

    # -- classification -------------------------------------------------
    def _parse(self, text: str) -> None:
        for no, stmt in logical_lines(text):
            head = stmt.split(",", 1)[0].split()[0].strip().rstrip(":").lower() if stmt.split() else ""
            if head == "return":
                break
            if self._command(head, stmt, no):
                continue
            m = _RE_LINE.match(stmt)
            if m:
                name = m.group(1).upper()
                self.lines[name] = self._line_entries(name, m.group(2))
                self.line_line[name] = no
                continue
            m = _RE_PARAM.match(stmt)
            if m and ":" not in m.group(1) and not _RE_ELEMENT_HEAD.match(stmt):
                self._add_param(m.group(1), m.group(2).strip(), no)
                continue
            m = _RE_ELEMENT.match(stmt)
            if m:
                name = m.group(1).upper()
                self.elems[name] = _ElementDef(name, m.group(2).lower(),
                                               parse_attributes(m.group(3) or ""), no)
                continue
            self.warnings.append(f"MAD8 line {no}: unrecognised statement {stmt[:70]!r}")
            self.rep.lossy("UNRECOGNISED_STATEMENT",
                           f"statement not understood by the MAD8 reader: {stmt[:70]!r}",
                           element=None, kind=None, line=no)

    def _command(self, head: str, stmt: str, no: int) -> bool:
        """True when *stmt* is a command with no lattice content (kept in ``meta``)."""
        if head == "beam":
            self.beam = parse_attributes(stmt.partition(",")[2])
            self.commands.append(stmt)
            return True
        if head == "title":
            m = _RE_TITLE.match(stmt)
            if m:
                self.title = (m.group(1) or m.group(2) or (m.group(3) or "").strip()) or None
            self.commands.append(stmt)
            return True
        if head == "use":
            body = stmt.partition(",")[2] or stmt.partition(" ")[2]
            for part in split_top_level(body):
                key, sep, val = part.partition("=")
                if sep and key.strip().lower() not in ("period", "line", "sequence"):
                    continue                       # USE, SUPERPOSE=… and friends
                target = (val if sep else key).strip().strip('"').upper()
                if target:
                    self.use = target
                    break
            self.commands.append(stmt)
            return True
        if head == "call":
            self.rep.dropped("CALL_NOT_FOLLOWED",
                             f"MAD8 CALL is not followed by the reader: {stmt[:70]!r}",
                             element=None, kind=None, line=no)
            self.commands.append(stmt)
            return True
        if head in IGNORED_COMMANDS and not _RE_ELEMENT_HEAD.match(stmt) and ":" not in stmt.split(",")[0]:
            self.commands.append(stmt)
            return True
        return False

    def _add_param(self, name: str, text: str, no: int) -> None:
        key = name.upper()
        if "'" in key:
            # MAD8 writes chromaticities as QX' / QY'; the name cannot appear in any
            # expression a parser can read, so it is recorded and skipped.
            self.unparseable[key] = text
            self.rep.lossy("UNPARSEABLE_IDENTIFIER",
                           f"parameter {name!r} carries an apostrophe; recorded in "
                           "meta['mad8_unparseable_params'] and skipped",
                           element=None, kind=None, line=no, value=text)
            return
        self.params[key] = text
        self.param_line[key] = no

    def _line_entries(self, owner: str, body: str) -> list[str]:
        """Entry list of ``LINE=(…)``; anonymous ``(A,B)`` groups become real sub-lines."""
        entries: list[str] = []
        for raw in split_top_level(body):
            entry = raw.strip()
            if not entry:
                continue
            rep, neg, inner = _entry_parts(entry)
            if inner.startswith("("):
                self._anon += 1
                sub = f"{owner}__SUB{self._anon}"
                self.lines[sub] = self._line_entries(sub, inner[1:-1])
                prefix = f"{rep}*" if rep != 1 else ""
                entries.append(f"{prefix}{'-' if neg else ''}{sub}")
            else:
                entries.append(entry.upper())
        return entries


def _entry_parts(entry: str) -> tuple[int, bool, str]:
    """``2*-CELL`` → ``(2, True, 'CELL')``; ``-CELL`` → ``(1, True, 'CELL')``."""
    e = entry.strip()
    m = _RE_REPEAT.match(e)
    if m:
        return int(m.group(1)), m.group(2) == "-", m.group(3).strip()
    if e.startswith("-"):
        return 1, True, e[1:].strip()
    return 1, False, e


# ---------------------------------------------------------------------------
# Lazy expression resolution
# ---------------------------------------------------------------------------
class _Scope(Mapping):
    """File parameters as a lazy mapping, so a deck's ``E := …`` shadows the built-in
    constant ``e`` at *every* nesting level (:class:`lattix.ir.expr.LazyResolver` consults
    the constants first once it recurses)."""

    def __init__(self, resolver: _Resolver) -> None:
        self._r = resolver

    def __getitem__(self, key: str) -> float:
        if not isinstance(key, str) or key.upper() not in self._r.deck.params:
            raise KeyError(key)
        return self._r.param(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key.upper() in self._r.deck.params

    def __iter__(self):
        return iter(self._r.deck.params)

    def __len__(self) -> int:
        return len(self._r.deck.params)


class _Resolver:
    """HELIX ``_Mad8File.resolve`` on lattix's :func:`lattix.ir.expr.evaluate`: memoised,
    cycle-detecting, with ``NAME[ATTR]`` element-attribute references."""

    def __init__(self, deck: _Deck) -> None:
        self.deck = deck
        self.scope = _Scope(self)
        self._params: dict[str, float] = {}
        self._attrs: dict[tuple[str, str], float] = {}
        self._busy: set[str] = set()

    # public -------------------------------------------------------------
    def eval(self, text: str) -> float:
        return evaluate(text, self.scope, self)

    def param(self, name: str) -> float:
        key = name.upper()
        if key in self._params:
            return self._params[key]
        text = self.deck.params.get(key)
        if text is None:
            raise ExpressionError(f"unknown MAD8 identifier {name!r}")
        token = f"param:{key}"
        if token in self._busy:
            raise ExpressionError(f"circular definition of {name!r}")
        self._busy.add(token)
        try:
            value = self.eval(text)
        finally:
            self._busy.discard(token)
        self._params[key] = value
        return value

    def attr(self, ename: str, attr: str) -> float:
        key = (ename.upper(), attr.lower())
        if key in self._attrs:
            return self._attrs[key]
        el = self.deck.elems.get(key[0])
        if el is None:
            raise ExpressionError(f"{ename}[{attr}] — unknown MAD8 element")
        raw = self.inherited_attrs(el).get(key[1])
        if raw is None or raw is True:
            value = 0.0
        else:
            token = f"attr:{key[0]}[{key[1]}]"
            if token in self._busy:
                raise ExpressionError(f"circular reference {ename}[{attr}]")
            self._busy.add(token)
            try:
                value = self.eval(str(raw))
            finally:
                self._busy.discard(token)
        self._attrs[key] = value
        return value

    def __call__(self, token: str) -> float:
        if "[" in token and token.endswith("]"):
            head, _, tail = token.partition("[")
            return self.attr(head, tail[:-1])
        return self.param(token)

    # element attribute inheritance --------------------------------------
    def inherited_attrs(self, el: _ElementDef, _depth: int = 0) -> dict[str, str | bool]:
        parent = self.deck.elems.get(el.type.upper())
        if parent is None or parent is el or _depth > 16:
            return el.attrs
        merged = dict(self.inherited_attrs(parent, _depth + 1))
        merged.update(el.attrs)
        return merged

    def resolved_type(self, el: _ElementDef, _depth: int = 0) -> str:
        parent = self.deck.elems.get(el.type.upper())
        if parent is None or parent is el or _depth > 16:
            return el.type
        return self.resolved_type(parent, _depth + 1)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "mad8"

    def read(self, path: Path, *, strict: bool = False, brho: float | None = None,
             species: str | Species | None = None, auto_periods: bool = True,
             frequency_Hz: float | None = None, keep_expressions: bool = True,
             use: str | None = None, **_ignored) -> tuple[Lattice, FidelityReport]:
        """Parse a MAD8 flat deck.

        ``brho`` (T·m) overrides the deck's rigidity; ``species`` is the fallback when the
        deck has no ``BEAM`` statement; ``auto_periods`` inserts the LATTICE brackets the
        LINE hierarchy implies; ``use`` overrides the root line.  In strict mode the first
        LOSSY/DROPPED entry raises :class:`~lattix.fidelity.TranslationError`.
        """
        path = Path(path)
        text = path.read_text(encoding="latin-1", errors="replace")
        tag = parse_reference_tag(text)
        species_tagged = False
        if species is None:
            if tag is not None:
                species, species_tagged = tag.species, True   # written by lattix: the species travels in a tag
            else:
                species = "h-"                                  # the PIP-II lineage default (see the docstring)
        rep = FidelityReport(source_format="mad8", source_file=str(path))
        warnings: list[str] = []
        deck = _Deck(text, rep, warnings)
        self._deck = deck
        self._rep = rep
        self._path = path
        self._warnings = warnings
        self._strict = strict
        self._keep = keep_expressions
        self._res = _Resolver(deck)
        self._tags = parse_tags(text)

        ref = self._reference(brho, species, frequency_Hz)

        if species_tagged:

            rep.equivalent("REFERENCE_FROM_TAG", f"species {ref.species.name!r} taken from the deck's "

                           "'! lattix: reference' tag", element=None, kind=None)
        self._brho = ref.brho_signed

        root = self._root_line(use)
        lat = Lattice(name=root, reference=ref, warnings=warnings)
        self._lattice = lat
        self._build_lines(lat, root)
        lat.use = root
        lat.variables = self._variables()
        lat.meta.update({
            "source_format": "mad8",
            "mad8_root_line": root,
            "mad8_commands": list(deck.commands),
            "mad8_defined_elements": len(deck.elems),
            "mad8_defined_lines": len(deck.lines),
        })
        if deck.title is not None:
            lat.meta["mad8_title"] = deck.title
            lat.meta.setdefault("madx_title", deck.title)
        if deck.unparseable:
            lat.meta["mad8_unparseable_params"] = dict(deck.unparseable)

        periods: list[dict] = []
        if auto_periods and root in lat.lines:
            periods = self._declare_periods(lat, root)
        lat.meta["mad8_periods"] = periods

        mode = _energy_mode_tag(text)
        if mode in ENERGY_MODES:
            # written by lattix: undo its normalization with the same energy walk
            if mode == "delta":
                undo_phase_slip(lat, rep)
            restore_energy_mode(lat, rep, mode)
        rep.raise_if(strict)
        return lat, rep

    # -- reference particle ------------------------------------------------
    def _reference(self, brho: float | None, species: str | Species | None,
                   frequency_Hz: float | None) -> ReferenceParticle:
        deck, rep = self._deck, self._rep
        sp, beam_ke = self._species_from_beam(species)

        source = "argument"
        brho_abs: float | None = None
        if brho is not None:
            brho_abs = abs(float(brho))
        elif "BRHO" in deck.params:
            try:
                brho_abs = abs(self._res.param("BRHO"))
                source = "BRHO parameter"
            except ExpressionError as exc:
                raise ValueError(f"MAD8: BRHO := {deck.params['BRHO']!r} cannot be evaluated ({exc})") from exc
        if brho_abs is not None:
            ke = ReferenceParticle.from_brho(sp, brho_abs).kinetic_energy_eV
            rep.equivalent("RIGIDITY_FROM_BRHO",
                           f"rigidity taken from the {source} (Bρ = {brho_abs:.6g} T·m) → "
                           f"{sp.name} at {ke / 1e6:.4f} MeV kinetic",
                           element=None, kind=None, brho=brho_abs,
                           kinetic_energy_eV=ke, species=sp.name)
            if beam_ke is not None:
                beam_brho = ReferenceParticle(species=sp, kinetic_energy_eV=beam_ke).brho_abs
                if abs(beam_brho - brho_abs) > _TOL_REL * max(beam_brho, 1.0):
                    rep.equivalent("RIGIDITY_CONFLICT",
                                   f"the BEAM statement implies Bρ = {beam_brho:.6g} T·m but the "
                                   f"{source} gives {brho_abs:.6g} T·m; the {source} wins",
                                   element=None, kind=None, brho_beam=beam_brho, brho_used=brho_abs)
            return ReferenceParticle(species=sp, kinetic_energy_eV=ke,
                                     rf_frequency_Hz=frequency_Hz)
        if beam_ke is not None:
            return ReferenceParticle(species=sp, kinetic_energy_eV=beam_ke,
                                     rf_frequency_Hz=frequency_Hz)
        raise ValueError(
            "MAD8: no rigidity available — supply it as read(..., brho=<T·m>), or declare "
            "`BRHO := <T·m>` in the file, or add a `BEAM, PARTICLE=…, ENERGY=…` statement.  "
            "Refusing to guess: a wrong Bρ silently mis-scales every magnet.")

    def _species_from_beam(self, fallback: str | Species | None) -> tuple[Species, float | None]:
        """(species, kinetic energy from BEAM or None).  The deck's BEAM wins."""
        deck, rep = self._deck, self._rep
        default = (fallback if isinstance(fallback, Species)
                   else species_by_name(fallback or "h-"))
        if not deck.beam:
            return default, None

        raw_name = str(deck.beam.get("particle", "") or "").strip().strip('"').lower()
        mass_eV = self._beam_number("mass")
        charge = self._beam_number("charge")
        sp: Species | None = None
        if raw_name:
            key = _PARTICLE_ALIAS.get(raw_name, raw_name)
            try:
                sp = species_by_name(key)
            except KeyError:
                sp = None
        if mass_eV is not None:
            mass_eV *= 1e9
            q = int(round(charge)) if charge is not None else (sp.charge if sp else 1)
            if sp is None or abs(sp.mass_eV - mass_eV) > _TOL_REL * max(mass_eV, 1.0) or sp.charge != q:
                sp = next((c for c in SPECIES.values()
                           if c.charge == q and abs(c.mass_eV - mass_eV) <= _TOL_REL * max(mass_eV, 1.0)),
                          None)
                if sp is None:
                    sp = Species(name=raw_name or "ion", mass_eV=mass_eV, charge=q)
                    self._warnings.append(
                        f"MAD8 BEAM mass={mass_eV / 1e9} GeV charge={q} matches no known species "
                        "— kept as a custom Species")
        if sp is None:
            sp = default
            self._warnings.append(
                f"MAD8 BEAM particle={raw_name!r} is not a species lattix knows — using {sp.name}")
            rep.equivalent("SPECIES_ASSUMED",
                           f"BEAM particle={raw_name!r} unknown; {sp.name} assumed",
                           element=None, kind=None, particle=raw_name, species=sp.name)
        elif charge is not None and int(round(charge)) != sp.charge:
            self._warnings.append(
                f"MAD8 BEAM charge={charge} disagrees with {sp.name} (q={sp.charge}); "
                "the species charge is used")

        ke: float | None = None
        energy = self._beam_number("energy")
        pc = self._beam_number("pc")
        gamma = self._beam_number("gamma")
        ekin = self._beam_number("ekin")
        if energy is not None:
            ke = energy * 1e9 - sp.mass_eV
        elif pc is not None:
            ke = math.hypot(pc * 1e9, sp.mass_eV) - sp.mass_eV
        elif gamma is not None:
            ke = (gamma - 1.0) * sp.mass_eV
        elif ekin is not None:
            ke = ekin * 1e9
        if ke is not None and ke <= 0:
            self._warnings.append(f"MAD8 BEAM gives a non-positive kinetic energy ({ke} eV)")
        return sp, ke

    def _beam_number(self, key: str) -> float | None:
        raw = self._deck.beam.get(key)
        if raw is None or raw is True:
            return None
        try:
            return self._res.eval(str(raw))
        except ExpressionError:
            self._warnings.append(f"MAD8 BEAM {key}={raw!r} could not be evaluated — ignored")
            return None

    # -- variables ----------------------------------------------------------
    def _variables(self) -> dict[str, Variable]:
        out: dict[str, Variable] = {}
        for name, text in self._deck.params.items():
            try:
                value = self._res.param(name)
            except ExpressionError as exc:
                self._rep.lossy("UNRESOLVED_VARIABLE",
                                f"parameter {name} := {text!r} could not be evaluated ({exc})",
                                element=None, kind=None, line=self._deck.param_line.get(name),
                                variable=name)
                continue
            expr = (Expression(text=text, deferred=True, dialect="infix")
                    if self._keep and not _looks_numeric(text) else None)
            # MAD8 is case insensitive; the IR keeps variables lower case like the MAD-X
            # reader does, so a knob survives mad8 -> IR -> MAD-X (whose identifiers must
            # be lower case).  The MAD8 writer upper-cases them again on the way out.
            out[name.lower()] = Variable(value=value, expression=expr)
        return out

    # -- root line -----------------------------------------------------------
    def _root_line(self, requested: str | None) -> str:
        deck = self._deck
        if not deck.lines:
            raise ValueError("MAD8: the deck defines no LINE")
        for candidate in (requested, deck.use):
            if candidate and candidate.upper() in deck.lines:
                return candidate.upper()
            if candidate:
                raise KeyError(f"MAD8: no LINE named {candidate!r}; known: {sorted(deck.lines)}")
        referenced: set[str] = set()
        for entries in deck.lines.values():
            for entry in entries:
                referenced.add(_entry_parts(entry)[2].upper())
        roots = [n for n in deck.lines if n not in referenced]
        if not roots:
            raise ValueError("MAD8: no top-level LINE found (every line is referenced by another)")
        if len(roots) == 1:
            return roots[0]
        sized = sorted(((self._expansion_size(r), r) for r in roots), reverse=True)
        msg = (f"MAD8: {len(roots)} top-level LINEs; using the largest, {sized[0][1]!r} "
               f"({sized[0][0]} entries); others: {[r for _, r in sized[1:]]}")
        self._warnings.append(msg)
        # EQUIVALENT, not LOSSY: a SAVELINE deck always carries unreferenced fragments
        # (the PIP-II BTL has 53 leftover steerer wrappers), the rule is deterministic and
        # the caller can override it with ``use=``.  Nothing physical is chosen away.
        self._rep.equivalent("AMBIGUOUS_ROOT_LINE", msg, element=None, kind=None,
                             chosen=sized[0][1], roots=[r for _, r in sized])
        return sized[0][1]

    def _expansion_size(self, name: str, depth: int = 0) -> int:
        if depth > 64 or name not in self._deck.lines:
            return 1
        total = 0
        for entry in self._deck.lines[name]:
            rep, _neg, ref = _entry_parts(entry)
            total += rep * self._expansion_size(ref.upper(), depth + 1)
        return total

    # -- lines and elements ---------------------------------------------------
    def _build_lines(self, lat: Lattice, root: str) -> None:
        """Register every line reachable from *root* plus the elements they use."""
        pending = [root]
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            items: list[LineItem] = []
            for entry in self._deck.lines[name]:
                rep, neg, ref = _entry_parts(entry)
                ref = ref.upper()
                if ref in self._deck.lines:
                    pending.append(ref)
                elif ref in self._deck.elems:
                    self._ensure_element(lat, ref)
                else:
                    msg = f"MAD8: {ref!r} is referenced by LINE {name} but never defined"
                    self._warnings.append(msg + " — skipped")
                    self._rep.dropped("UNDEFINED_REFERENCE", msg, element=ref, kind=None,
                                      line=self._deck.line_line.get(name))
                    continue
                items.append(LineItem(ref=ref, repeat=rep, reverse=neg))
            lat.lines[name] = Line(name=name, items=items)

    def _ensure_element(self, lat: Lattice, name: str) -> None:
        if name in lat.elements:
            return
        el = self._build_element(name)
        lat.elements[name] = el

    def _attrs(self, ename: str) -> dict[str, Any]:
        """Element attributes with every value resolved (unresolvable → 0 + LOSSY)."""
        el = self._deck.elems[ename]
        raw = self._res.inherited_attrs(el)
        out: dict[str, Any] = {}
        for key, value in raw.items():
            if value is True:
                out[key] = True
                continue
            if key in _STRING_ATTRS:
                out[key] = str(value).strip().strip('"')
                continue
            try:
                out[key] = self._res.eval(str(value))
            except ExpressionError as exc:
                msg = f"MAD8: {ename}.{key} = {value!r} could not be evaluated ({exc})"
                self._warnings.append(msg + " — treated as 0")
                self._rep.lossy("UNRESOLVED_ATTRIBUTE", msg, element=ename, kind=None,
                                line=el.line, attr=key, expression=str(value))
                out[key] = 0.0
        return out

    def _raw(self, ename: str, attr: str) -> str | None:
        """The source text of an attribute, when it is an expression worth keeping."""
        if not self._keep:
            return None
        value = self._res.inherited_attrs(self._deck.elems[ename]).get(attr)
        if value is None or value is True or _looks_numeric(str(value)):
            return None
        return str(value).strip()

    def _expr(self, el: Element, ename: str, attr: str, path: str) -> None:
        text = self._raw(ename, attr)
        if text is None:
            return
        el.expressions[path] = Expression(text=text, deferred=True, dialect="infix")
        el.native.setdefault("mad8", {})[f"{attr}_expr"] = text

    @staticmethod
    def _tilt(attrs: dict[str, Any], natural: float) -> float:
        value = attrs.get("tilt", 0.0)
        if value is True:
            return natural
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _build_element(self, ename: str) -> Element:
        deck, rep = self._deck, self._rep
        defn = deck.elems[ename]
        etype = self._res.resolved_type(defn)
        attrs = self._attrs(ename)
        handler = getattr(self, f"_conv_{etype}", None)
        n_before = len(rep.entries)
        if handler is None or etype not in ELEMENT_TYPES:
            el: Element = Marker(name=ename, length=float(attrs.get("l", 0.0) or 0.0))
            el.native["mad8"] = {"type": etype, "attrs": {k: v for k, v in attrs.items()
                                                          if not isinstance(v, bool)}}
            rep.dropped("UNSUPPORTED_MAD8_TYPE",
                        f"MAD8 type {etype!r} has no IR mapping; kept as a marker of the same length",
                        element=ename, kind="Marker", line=defn.line, mad8_type=etype)
        else:
            el = handler(ename, attrs)
            el.native.setdefault("mad8", {})["type"] = etype
            if len(rep.entries) == n_before:
                rep.exact(ename, el.kind)
        tag = self._tags.get(ename, {})
        if tag.get("kind") == "ReferenceChange" and el.kind == "Marker":
            fields = {f: float(tag[f]) for f in ("dE_ref_eV", "energy_eV", "dtime_s", "dphase_rad") if f in tag}
            el = ReferenceChange(name=ename, **fields)
            rep.entries = [e for e in rep.entries if e.element != ename]
            rep.equivalent("REFCHANGE_FROM_TAG",
                           "reference change restored from the marker's lattix tag (MAD8 itself ignores it)",
                           element=ename, kind="ReferenceChange", line=defn.line, **fields)
        el.provenance = Provenance(format="mad8", file=str(self._path), line=defn.line,
                                   original_name=tag.get("name", ename),
                                   original_type=(tag.get("type") if tag else etype.upper()))
        self._apply_aperture(el, ename, attrs)
        self._expr(el, ename, "l", "length")
        return el

    def _apply_aperture(self, el: Element, ename: str, attrs: dict[str, Any]) -> None:
        if el.kind == "Collimator":
            return
        radius = attrs.get("aperture")
        if isinstance(radius, (int, float)) and radius:
            el.aperture = ApertureP.circle(float(radius))

    # -- per-type conversions ------------------------------------------------
    @staticmethod
    def _len(attrs: dict[str, Any]) -> float:
        value = attrs.get("l", 0.0)
        return 0.0 if value is True else float(value or 0.0)

    def _conv_drift(self, name, attrs):
        return Drift(name=name, length=self._len(attrs))

    def _conv_quadrupole(self, name, attrs):
        el = Quadrupole(name=name, length=self._len(attrs))
        el.multipole.Bn[1] = float(attrs.get("k1", 0.0) or 0.0) * self._brho
        tilt = self._tilt(attrs, _NATURAL_TILT["quadrupole"])
        if tilt:
            el.multipole.tilt[1] = tilt
        self._expr(el, name, "k1", "multipole.Bn[1]")
        self._expr(el, name, "tilt", "multipole.tilt[1]")
        return el

    def _conv_sextupole(self, name, attrs):
        return self._thick_multipole(name, attrs, Sextupole, 2, "k2", "sextupole")

    def _conv_octupole(self, name, attrs):
        return self._thick_multipole(name, attrs, Octupole, 3, "k3", "octupole")

    def _thick_multipole(self, name, attrs, cls, order, key, tilt_key):
        el = cls(name=name, length=self._len(attrs))
        el.multipole.Bn[order] = float(attrs.get(key, 0.0) or 0.0) * self._brho
        tilt = self._tilt(attrs, _NATURAL_TILT[tilt_key])
        if tilt:
            el.multipole.tilt[order] = tilt
        self._expr(el, name, key, f"multipole.Bn[{order}]")
        self._expr(el, name, "tilt", f"multipole.tilt[{order}]")
        return el

    def _conv_multipole(self, name, attrs):
        """MAD8 spells a thin multipole ``K0L … K20L`` with per-order tilts ``T0 … T20``
        (MAD-X's ``knl={…}`` vectors are accepted too, for MAD-X-flavoured files)."""
        el = Multipole(name=name, length=0.0)
        for key, value in attrs.items():
            m = re.fullmatch(r"k(\d+)l", key)
            if m and isinstance(value, (int, float)) and value:
                el.multipole.BnL[int(m.group(1))] = float(value) * self._brho
            m = re.fullmatch(r"k(\d+)sl", key)
            if m and isinstance(value, (int, float)) and value:
                el.multipole.BsL[int(m.group(1))] = float(value) * self._brho
            m = re.fullmatch(r"t(\d+)", key)
            if m:
                order = int(m.group(1))
                tilt = value if not isinstance(value, bool) else math.pi / (2 * (order + 1))
                if tilt:
                    el.multipole.tilt[order] = float(tilt)
        for vec_key, table in (("knl", el.multipole.BnL), ("ksl", el.multipole.BsL)):
            raw = self._res.inherited_attrs(self._deck.elems[name]).get(vec_key)
            if isinstance(raw, str) and raw.strip().startswith("{"):
                for order, part in enumerate(split_top_level(raw.strip()[1:-1])):
                    try:
                        value = self._res.eval(part)
                    except ExpressionError:
                        continue
                    if value:
                        table[order] = value * self._brho
        lrad = attrs.get("lrad")
        if isinstance(lrad, (int, float)) and lrad:
            el.native.setdefault("mad8", {})["lrad"] = float(lrad)
        return el

    def _conv_sbend(self, name, attrs):
        return self._bend(name, attrs, rect=False)

    def _conv_rbend(self, name, attrs):
        return self._bend(name, attrs, rect=True)

    def _bend(self, name, attrs, *, rect: bool):
        # MAD8 has no OPTION, RBARC: L is the ARC length for both SBEND and RBEND
        # (verified on the PIP-II BTL anchor, see the module docstring).
        angle = float(attrs.get("angle", 0.0) or 0.0)
        half = angle / 2.0 if rect else 0.0
        fint = float(attrs.get("fint", 0.0) or 0.0)
        el = Bend(name=name, length=self._len(attrs))
        el.bend = BendP(
            angle=angle,
            e1=float(attrs.get("e1", 0.0) or 0.0) + half,
            e2=float(attrs.get("e2", 0.0) or 0.0) + half,
            edge_int1=fint,
            hgap=float(attrs.get("hgap", 0.0) or 0.0),
            tilt_ref=self._tilt(attrs, _NATURAL_TILT["sbend"]),
            rect=rect,
        )
        for order, key in ((1, "k1"), (2, "k2"), (3, "k3")):
            value = attrs.get(key)
            if isinstance(value, (int, float)) and value:
                el.multipole.Bn[order] = float(value) * self._brho
                self._expr(el, name, key, f"multipole.Bn[{order}]")
        self._expr(el, name, "angle", "bend.angle")
        self._expr(el, name, "e1", "bend.e1")
        self._expr(el, name, "e2", "bend.e2")
        extra = {k: attrs[k] for k in ("h1", "h2", "k1s", "k2s", "k3s", "ks")
                 if isinstance(attrs.get(k), (int, float)) and attrs[k]}
        if extra:
            el.native.setdefault("mad8", {}).update(extra)
            self._rep.lossy("BEND_ATTR_DROPPED",
                            f"bend attributes not modelled by the IR: {sorted(extra)}",
                            element=name, kind="Bend", **extra)
        return el

    def _conv_solenoid(self, name, attrs):
        el = Solenoid(name=name, length=self._len(attrs))
        el.solenoid.Bsol_T = float(attrs.get("ks", 0.0) or 0.0) * self._brho
        self._expr(el, name, "ks", "solenoid.Bsol_T")
        return el

    def _conv_rfcavity(self, name, attrs):
        el = RFCavity(name=name, length=self._len(attrs))
        el.rf.voltage_V = float(attrs.get("volt", 0.0) or 0.0) * 1e6        # MAD8 VOLT is MV
        el.rf.phase_rad = phase_from_madx_lag(float(attrs.get("lag", 0.0) or 0.0))
        freq = float(attrs.get("freq", 0.0) or 0.0)                          # MAD8 FREQ is MHz
        el.rf.frequency_Hz = freq * 1e6 if freq else None
        if el.length:
            el.rf.L_active_m = el.length
        harmon = attrs.get("harmon")
        if isinstance(harmon, (int, float)) and harmon:
            el.native.setdefault("mad8", {})["harmon"] = int(harmon)
        self._expr(el, name, "volt", "rf.voltage_V")
        self._expr(el, name, "lag", "rf.phase_rad")
        self._expr(el, name, "freq", "rf.frequency_Hz")
        return el

    _conv_twcavity = _conv_rfcavity

    def _conv_kicker(self, name, attrs):
        el = Kicker(name=name, length=self._len(attrs),
                    hkick=float(attrs.get("hkick", 0.0) or 0.0),
                    vkick=float(attrs.get("vkick", 0.0) or 0.0))
        self._expr(el, name, "hkick", "hkick")
        self._expr(el, name, "vkick", "vkick")
        self._kicker_tilt(el, name, attrs)
        return el

    def _conv_hkicker(self, name, attrs):
        el = Kicker(name=name, length=self._len(attrs),
                    hkick=float(attrs.get("kick", 0.0) or 0.0))
        self._expr(el, name, "kick", "hkick")
        self._kicker_tilt(el, name, attrs)
        return el

    def _conv_vkicker(self, name, attrs):
        el = Kicker(name=name, length=self._len(attrs),
                    vkick=float(attrs.get("kick", 0.0) or 0.0))
        self._expr(el, name, "kick", "vkick")
        self._kicker_tilt(el, name, attrs)
        return el

    def _kicker_tilt(self, el: Kicker, name: str, attrs: dict[str, Any]) -> None:
        tilt = self._tilt(attrs, math.pi / 2)
        if tilt:
            el.native.setdefault("mad8", {})["tilt"] = tilt
            self._rep.lossy("KICKER_TILT_DROPPED",
                            f"TILT={tilt:.6g} rad on a kicker is not modelled by the IR "
                            "(hkick/vkick are lab-frame deflections)",
                            element=name, kind="Kicker", tilt=tilt)

    def _conv_elseparator(self, name, attrs):
        """MAD8 ELSEPARATOR: a vertical electrostatic deflection ``E`` [MV/m].  The IR's
        kicker holds an angle, so the conversion needs the reference momentum — energy
        dependent, hence EQUIVALENT rather than EXACT."""
        length = self._len(attrs)
        field_MV_per_m = float(attrs.get("e", 0.0) or 0.0)
        ref = self._lattice.reference if getattr(self, "_lattice", None) else None
        pc_eV = ref.pc_eV if ref is not None else 0.0
        beta = ref.beta if ref is not None else 1.0
        kick = 0.0
        if pc_eV and length:
            sign = 1.0 if self._brho >= 0 else -1.0
            kick = sign * field_MV_per_m * 1e6 * length / (beta * pc_eV)
        el = Kicker(name=name, length=length, vkick=kick, electric=True)
        el.native.setdefault("mad8", {})["e"] = field_MV_per_m
        self._rep.equivalent("ESEPARATOR_AS_KICKER",
                             f"ELSEPARATOR E={field_MV_per_m:g} MV/m converted to a "
                             f"{kick:.6g} rad vertical kick at the entrance momentum",
                             element=name, kind="Kicker", e_MV_per_m=field_MV_per_m, vkick=kick)
        return el

    def _conv_monitor(self, name, attrs):
        etype = self._res.resolved_type(self._deck.elems[name])
        return Instrument(name=name, length=self._len(attrs),
                          family=_MONITOR_FAMILY.get(etype, "MONITOR"))

    _conv_hmonitor = _conv_monitor
    _conv_vmonitor = _conv_monitor
    _conv_instrument = _conv_monitor
    _conv_blmonitor = _conv_monitor
    _conv_slmonitor = _conv_monitor
    _conv_wire = _conv_monitor
    _conv_profile = _conv_monitor

    def _conv_rcollimator(self, name, attrs):
        el = Collimator(name=name, length=self._len(attrs))
        xs = float(attrs.get("xsize", 0.0) or 0.0)
        ys = float(attrs.get("ysize", 0.0) or 0.0)
        if xs or ys:
            el.aperture = ApertureP.rect(xs, ys or xs)
        return el

    def _conv_ecollimator(self, name, attrs):
        el = Collimator(name=name, length=self._len(attrs))
        xs = float(attrs.get("xsize", 0.0) or 0.0)
        ys = float(attrs.get("ysize", 0.0) or 0.0)
        if xs or ys:
            el.aperture = ApertureP(shape="ELLIPTICAL", x_limits=(-xs, xs),
                                    y_limits=(-(ys or xs), ys or xs))
        return el

    def _conv_marker(self, name, attrs):
        return Marker(name=name, length=self._len(attrs))

    def _conv_srot(self, name, attrs):
        return Patch(name=name, tilt=float(attrs.get("angle", 0.0) or 0.0))

    def _conv_yrot(self, name, attrs):
        return Patch(name=name, y_rot=float(attrs.get("angle", 0.0) or 0.0))

    # -- periodicity -----------------------------------------------------------
    def _declare_periods(self, lat: Lattice, root: str) -> list[dict]:
        """Port of HELIX ``_declare_periods``: bracket maximal runs of identical cells."""
        cells = _collect_cells(lat, root)
        if not cells:
            return []
        placed = lat.flatten(root)
        for cell in cells:
            segment = [p.element for p in placed[cell["i0"]:cell["i1"]]]
            cell["sig"] = _cell_signature(segment)
            cell["n_sig"] = sum(_significant_cards(e) for e in segment)

        declared: list[dict] = []
        by_parent: dict[str, list[dict]] = {}
        for cell in cells:
            by_parent.setdefault(cell["parent"], []).append(cell)
        for parent, group in by_parent.items():
            group.sort(key=lambda c: c["i0"])

            def repeats(cell, group=group) -> bool:
                if not any(item[0] in ("Q", "B") for item in cell["sig"]):
                    return False        # a wire-scanner wrapper LINE is not a transport period
                return sum(1 for other in group if _sig_equal(cell["sig"], other["sig"])) >= 2

            runs: list[list[dict]] = []
            for cell in group:
                if not repeats(cell):
                    continue
                last = runs[-1][-1] if runs else None
                if (last is not None and last["i1"] == cell["i0"]
                        and last["line"] == cell["line"]
                        and last["item"] + 1 == cell["item"]
                        and _sig_equal(last["sig"], cell["sig"])
                        and last["n_sig"] == cell["n_sig"]):
                    runs[-1].append(cell)
                else:
                    runs.append([cell])
            for run in runs:
                declared.append({
                    "cells": [c["name"] for c in run], "parent": parent,
                    "line": run[0]["line"], "item0": run[0]["item"], "item1": run[-1]["item"],
                    "n_repeats": len(run), "n_sig": run[0]["n_sig"],
                    "i0": run[0]["i0"], "i1": run[-1]["i1"],
                })

        declared.sort(key=lambda d: d["i0"])
        # insert back to front so the item indices of earlier runs stay valid
        for k, d in enumerate(reversed(declared)):
            index = len(declared) - k
            open_d = Directive(name=f"LATTICE_{index}", format="tracewin", card="LATTICE",
                               args=[str(d["n_sig"]), "0"], role="period_start",
                               provenance=Provenance(format="mad8", file=str(self._path),
                                                     original_type="LATTICE"))
            close_d = Directive(name=f"LATTICE_END_{index}", format="tracewin",
                                card="LATTICE_END", args=[], role="period_end",
                                provenance=Provenance(format="mad8", file=str(self._path),
                                                      original_type="LATTICE_END"))
            lat.elements[open_d.name] = open_d
            lat.elements[close_d.name] = close_d
            items = lat.lines[d["line"]].items
            items.insert(d["item1"] + 1, LineItem(ref=close_d.name))
            items.insert(d["item0"], LineItem(ref=open_d.name))
        return [{k: v for k, v in d.items() if k != "sig"} for d in declared]


# ---------------------------------------------------------------------------
# Periodicity helpers (HELIX _cell_signature / _is_significant on the IR)
# ---------------------------------------------------------------------------
#: 2 µm — HELIX's ``_SIG_TOL_MM`` in metres.
SIG_TOL_M = 2e-6


def _collect_cells(lat: Lattice, root: str) -> list[dict]:
    """LINE-valued grandchildren of the root with their flat index ranges.

    Depth 0 = the root's entries (machine sections), depth 1 = their LINE-valued
    entries = candidate periodic cells (HELIX ``parse_mad8.walk``).
    """
    cells: list[dict] = []
    counter = [0]
    used: dict[str, int] = {}

    def walk(lname: str, reverse: bool, depth: int, parent: str) -> None:
        if lname not in lat.lines:
            counter[0] += 1
            return
        used[lname] = used.get(lname, 0) + 1
        items = list(enumerate(lat.lines[lname].items))
        if reverse:
            items = list(reversed(items))
        for idx, item in items:
            for _ in range(item.repeat):
                child_rev = reverse ^ item.reverse
                if item.ref in lat.lines:
                    if depth == 1:
                        i0 = counter[0]
                        walk(item.ref, child_rev, depth + 1, parent)
                        cells.append({"name": item.ref, "parent": parent, "line": lname,
                                      "item": idx, "i0": i0, "i1": counter[0]})
                    else:
                        walk(item.ref, child_rev, depth + 1,
                             item.ref if depth == 0 else parent)
                else:
                    counter[0] += 1

    walk(root, False, 0, root)
    # a section line used more than once would get its brackets duplicated at every
    # occurrence, so those cells are not bracketable
    return [c for c in cells if used.get(c["line"], 0) == 1]


def _cell_signature(elements: list[Element]) -> tuple:
    """Transport signature: consecutive drift-like elements merged, non-transport skipped."""
    sig: list[list] = []

    def drift(length: float) -> None:
        if sig and sig[-1][0] == "D":
            sig[-1][1] += length
        else:
            sig.append(["D", length])

    for el in elements:
        kind = el.kind
        if kind == "Drift":
            drift(el.length)
        elif kind == "Kicker" and not (el.hkick or el.vkick):
            drift(el.length)             # a zero-strength steerer is a drift, as in HELIX
        elif kind == "Kicker":
            sig.append(["K", el.length, el.hkick, el.vkick])
        elif kind == "Quadrupole":
            sig.append(["Q", el.length, el.multipole.Bn.get(1, 0.0)])
        elif kind == "Bend":
            if el.bend.angle == 0.0:
                drift(el.length)         # a zero-angle placeholder RBEND is a drift
            else:
                sig.append(["B", el.length, el.bend.angle, el.bend.e1, el.bend.e2])
        elif kind == "Solenoid":
            sig.append(["S", el.length, el.solenoid.Bsol_T])
        elif el.length:
            drift(el.length)
    return tuple(tuple(s) for s in sig)


def _sig_equal(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x[0] != y[0] or any(abs(p - q) > SIG_TOL_M
                               for p, q in zip(x[1:], y[1:], strict=True)):
            return False
    return True


def _significant_cards(el: Element) -> int:
    """How many elements TraceWin counts for this IR element inside a ``LATTICE n`` cell.

    Per the TraceWin manual, ``DIAG_*``, ``APERTURE`` and markers are *not* counted; a
    zero-length drift carries no transport.  A ``Bend`` becomes EDGE + BEND + EDGE when it
    has pole faces (what :mod:`lattix.formats.tracewin.writer` emits), so it counts 3.
    """
    kind = el.kind
    if kind in ("Marker", "Instrument", "Directive", "Collimator", "Freq",
                "ReferenceChange", "Patch"):
        return 0
    if kind == "Bend":
        if el.bend.angle == 0.0:
            return 1 if el.length else 0
        return 3 if (el.bend.e1 or el.bend.e2 or el.bend.hgap) else 1
    if kind == "Drift" and not el.length:
        return 0
    return 1


def read(path: str | Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)


__all__ = ["ELEMENT_TYPES", "IGNORED_COMMANDS", "Reader", "logical_lines", "parse_attributes",
           "parse_tags", "read", "split_top_level", "strip_comment"]
