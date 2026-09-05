"""Bmad lattice reader (PLAN §6 task 2.2).

The parser is hand written (Cheetah's ``cheetah/converters/bmad.py`` was the
structural reference); Bmad's grammar is small but has enough quirks —
``call::file`` inside an element definition, brace blocks, bare attribute flags,
``-line`` reflections — that a token-level recursive descent is clearer than a
grammar file.

Syntax facts (``bmad/doc/lattice-file.tex`` §"Lattice File Syntax", all verified
against Bmad 20260828.0 on 2026-09-03):

* ``!`` starts a comment; ``;`` separates statements on one line; a statement
  continues onto the next line when the line ends with ``&`` or with one of
  ``, ( { [ =``, or when the next line *starts* with one of ``, ) } ] =``;
* Bmad is case insensitive (names are upper-cased internally); the reader works
  in lower case and keeps the source spelling in ``Provenance.original_name``;
* names are limited to 40 characters (measured: 41 fails, see
  :mod:`lattix.formats.bmad.naming`);
* an element definition is ``name: type, attr = value, …``; *type* may be a
  previously defined element, which makes the new element inherit its type and
  attributes (``quadrupole1: q0, l = 0.6, k1 = 1``);
* an attribute may appear with no value (``tilt``, ``t1``): that selects Bmad's
  own default, which for a multipole tilt is ``π/(2(n+1))`` — measured
  π/4 (quadrupole), π/6 (sextupole), π/8 (octupole);
* ``name[attr] = value`` overrides an attribute after the fact and is honoured
  wherever it appears in the file (measured: after ``use`` too).

Physics conventions measured with Tao (2026-09-03), not recalled:

* **rigidity** — ``k1``/``ks``/``kick`` are species-*independent* in Bmad
  exactly as in MAD-X (a ``#1H-`` deck and a proton deck with the same ``k1``
  give the same map), so the IR's lab field is ``Bn[1] = k1 · Bρ_signed`` with
  the *signed* rigidity, the same rule the MAD-X reader uses.  Bmad's own
  ``b1_gradient`` is ``k1 · p0c/c`` for every species and is therefore **not**
  the lab field for a negative species; the reader never uses it as one without
  saying so (``BMAD_FIELD_MASTER``).
* **reference energy follows the lattice** — Bmad's ``lcavity`` moves ``p0``, so
  the local ``Bρ`` differs section by section.  The reader therefore builds the
  IR in two passes: pass 1 stores the deck's normalized numbers in
  ``native["bmad"]``, pass 2 walks the lattice with
  :func:`lattix.ir.walk.propagate` and converts every normalized strength with
  the rigidity at *that element's own entrance*.
* **rbend** — a deck's ``l`` on an rbend is the **chord**; Bmad stores the arc
  ``L_chord·(θ/2)/sin(θ/2)`` and shifts the pole faces to the sector reference
  with ``e1 += θ/2``, ``e2 += θ/2`` (measured: ``rbend, l=0.8, angle=0.08,
  e1=0.01`` → ``l = 0.80021``, ``l_chord = 0.8``, ``e1 = 0.05``).  The IR always
  stores the arc length and sector-referenced pole faces with ``BendP.rect``
  recording that the source was rectangular.
* **lcavity** — ``phi0`` is in turns with ``dE = V·cos(2π·phi0)``, species
  independent (measured: ``phi0 = −1/12`` on a 1 MV cavity → ΔE = 866 025.40 eV),
  i.e. exactly :func:`lattix.ir.rf.phase_from_bmad_phi0`.
* **rfcavity** — a ring cavity.  Bmad's ``rfcavity`` uses
  ``dE = −q·V·sin(2π(φ_t − phi0))`` where ``lcavity`` uses
  ``+cos(2π(φ_t + phi0))``, i.e. a *reflection*: the manual writes the
  equivalence as ``0.25 − phi0`` (``bmad/doc/attributes.tex:2551``), so the IR
  synchronous phase is ``2π(0.25 − phi0)``.  Measured on 20260828.0: the
  reference particle gains no energy for any ``phi0`` and the reference energy
  does not change, so the IR sets ``dE_ref_eV = 0`` and records
  ``EQUIVALENT:RING_RFCAVITY``.  The writer never emits ``rfcavity``.
* **multipole** — ``k{n}l``/``k{n}sl`` are *exactly* MAD-X's ``knl[n]``/``ksl[n]``
  (verified: a Bmad ``multipole, k1l, k1sl, k2l, k2sl`` and the matching MAD-X
  ``multipole, knl={…}, ksl={…}`` agree to 0 in the transverse block; the same
  holds for ``ab_multipole, a1`` against MAD-X ``ksl``, with **no** sign flip —
  Bmad's own ``bmad_to_mad_sad_elegant`` does flip it, see
  :mod:`lattix.formats.bmad.writer`).  Bmad's ``ab_multipole`` uses
  ``b{n} = k{n}l/n!`` and ``a{n} = k{n}sl/n!`` (measured: ``k2l = 0.3`` →
  ``B2(equiv) = 0.15``), and ``t{n}`` rotates them as
  ``b_n = k_nl·cos((n+1)t_n) + k_nsl·sin((n+1)t_n)``,
  ``a_n = k_nsl·cos((n+1)t_n) − k_nl·sin((n+1)t_n)``.
* **``fint``/``fintx`` written as a bare flag mean 0.5**, absent means 0, and an
  absent ``fintx`` follows ``fint`` (Bmad's own MAD-compatibility rule, which is
  what the IR's ``BendP.edge_int2 = None`` records).
* **bends have no ``tilt``**: ``ref_tilt`` rotates the bend *and* the reference
  orbit (MAD-X's ``tilt`` → ``BendP.tilt_ref``) while ``roll`` is the
  misalignment (→ ``BodyShiftP.tilt``).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from lattix.fidelity import FidelityReport
from lattix.formats.bmad.naming import parse_tags
from lattix.ir.elements import (
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    FieldMap,
    Foil,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    Octupole,
    Patch,
    Provenance,
    Quadrupole,
    RFCavity,
    Sextupole,
    Solenoid,
    Taylor,
)
from lattix.ir.expr import Expression, ExpressionError, LazyResolver, evaluate
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name
from lattix.ir.rf import phase_from_bmad_phi0
from lattix.ir.walk import propagate

C_LIGHT = 299_792_458.0

#: Bmad species tokens → lattix species names.  ``#1H-`` is the H-1 atom with one
#: extra electron; plain ``H-`` would use the isotope-averaged hydrogen mass
#: (``lattix/oracles/bmad_worker.py`` uses the same table in the other direction).
SPECIES_FROM_BMAD: dict[str, str] = {
    "proton": "proton", "antiproton": "antiproton", "electron": "electron",
    "positron": "positron", "deuteron": "deuteron",
    "#1h-": "h-", "h-": "h-", "1h-": "h-", "h_minus": "h-",
}

#: statements that carry no physics for the IR.
_IGNORED_STATEMENTS = frozenset({
    "expand_lattice", "no_digested", "end_file", "return", "write_digested",
    "merge_elements", "combine_consecutive_elements", "debug_marker",
})

#: controller/structural definitions: kept as a Directive plus a LOSSY entry.
_CONTROLLER_TYPES = frozenset({"overlay", "group", "girder", "ramper", "feedback"})

#: Bmad element keys that become an :class:`Instrument` with this ``family``.
_INSTRUMENT_FAMILY = {"monitor": "BPM", "instrument": "INSTRUMENT", "detector": "DETECTOR"}

_NAME = r"[A-Za-z][A-Za-z0-9_.#\\]*"
_RE_ATTR_SET = re.compile(rf"^({_NAME})\s*\[\s*([A-Za-z0-9_]+)\s*\]\s*=\s*(.*)$", re.DOTALL)
_RE_DEF = re.compile(rf"^({_NAME})\s*:\s*(.*)$", re.DOTALL)
_RE_VAR = re.compile(rf"^({_NAME})\s*(:?=)\s*(.*)$", re.DOTALL)
#: ``name := expr`` is MAD-compatible syntax for a constant (lattice-file.tex:1022);
#: it must be recognised before ``name: type`` or the ``:`` reads as a definition.
_RE_DEFERRED_VAR = re.compile(rf"^({_NAME})\s*:=\s*(.*)$", re.DOTALL)
_RE_TERM = re.compile(r"^\{\s*([A-Za-z0-9_]+)\s*:(.*)\|(.*)\}$", re.DOTALL)
_RE_MULT_KL = re.compile(r"^k(\d+)(s?)l$")
_RE_MULT_AB = re.compile(r"^([ab])(\d+)$")
_RE_MULT_T = re.compile(r"^t(\d+)$")


# ---------------------------------------------------------------------------
# lexing: a Bmad file -> a list of logical statements
# ---------------------------------------------------------------------------
class Statement:
    """One logical Bmad statement with its source position."""

    __slots__ = ("text", "file", "line")

    def __init__(self, text: str, file: str, line: int) -> None:
        self.text = text
        self.file = file
        self.line = line

    def __repr__(self) -> str:                       # pragma: no cover - debugging aid
        return f"Statement({self.text!r} @ {self.file}:{self.line})"


def strip_comment(line: str) -> str:
    """Drop a ``!`` comment, honouring double- and single-quoted strings."""
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "!":
            return line[:i]
    return line


#: characters that continue a statement onto the next line when they END a line
_END_CONT = ",({[=&"
#: characters that append a line to the previous one when they START a line
_BEGIN_CONT = ",)}]="


def logical_lines(text: str) -> list[tuple[str, int]]:
    """Join Bmad continuations; returns ``(joined_text, first_line_number)``."""
    raw = [(strip_comment(ln), i + 1) for i, ln in enumerate(text.splitlines())]
    out: list[tuple[str, int]] = []
    buf, start, pending = "", 0, False
    for content, lineno in raw:
        s = content.strip()
        if not s:
            continue
        if pending or (buf and s[0] in _BEGIN_CONT):
            buf = f"{buf} {s}" if buf else s
        else:
            if buf:
                out.append((buf, start))
            buf, start = s, lineno
        pending = bool(buf) and buf.rstrip()[-1:] in _END_CONT
        if pending and buf.rstrip().endswith("&"):
            buf = buf.rstrip()[:-1].rstrip()
    if buf:
        out.append((buf, start))
    return out


def split_statements(joined: str) -> list[str]:
    """Split on ``;`` outside quotes and brackets."""
    out, depth, quote, cur = [], 0, "", []
    for ch in joined:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == ";" and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def split_top(text: str, sep: str = ",") -> list[str]:
    """Split on *sep* at bracket depth 0, outside quotes."""
    out, depth, quote, cur = [], 0, "", []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


class _Def:
    """A raw element definition: type token plus attributes as source text."""

    __slots__ = ("name", "type", "attrs", "flags", "terms", "file", "line", "src_name")

    def __init__(self, name: str, type_: str, file: str, line: int) -> None:
        self.name = name.lower()
        self.src_name = name
        self.type = type_.lower()
        self.attrs: dict[str, str] = {}
        self.flags: set[str] = set()          # attributes given with no value
        self.terms: list[str] = []            # taylor ``{…}`` terms
        self.file = file
        self.line = line

    def inherit(self, parent: _Def) -> None:
        self.type = parent.type
        merged = dict(parent.attrs)
        merged.update(self.attrs)
        self.attrs = merged
        self.flags |= parent.flags
        self.terms = parent.terms + self.terms


class BmadSyntaxError(ValueError):
    pass


# ---------------------------------------------------------------------------
def _is_literal(text: str) -> bool:
    try:
        float(text.strip())
    except ValueError:
        return False
    return True


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "bmad"

    def read(self, path: Path, *, use: str | None = None, strict: bool = False,
             species: str | Species | None = None, kinetic_energy_eV: float | None = None,
             keep_expressions: bool = True) -> tuple[Lattice, FidelityReport]:
        path = Path(path).resolve()
        rep = FidelityReport(source_format="bmad", source_file=str(path))
        self._rep = rep
        self._warnings: list[str] = []
        self._defs: dict[str, _Def] = {}
        self._order: list[str] = []
        self._lines: dict[str, list[str]] = {}          # name -> raw item texts
        self._line_attrs: dict[str, dict[str, str]] = {}
        self._vars: dict[str, str] = {}                 # lower name -> source text
        self._var_order: list[str] = []
        self._params: dict[str, str] = {}
        self._beginning: dict[str, str] = {}
        self._particle_start: dict[str, str] = {}
        self._use: list[str] = []
        self._title: str | None = None
        self._anon = 0
        self._keep = keep_expressions

        text = path.read_text(encoding="utf-8", errors="replace")
        self._tags = parse_tags(text)
        self._parse_file(path, text, depth=0)

        resolver = self._make_resolver()
        root = self._root_line(use)
        self._merge_line_attrs(root)
        ref = self._reference(resolver, species, kinetic_energy_eV)
        lat = Lattice(name=root, reference=ref, warnings=self._warnings)
        lat.variables = self._variables(resolver)

        supers = self._build_elements(lat, resolver, str(path))
        self._build_lines(lat, root)
        if supers:
            self._resolve_superimpose(lat, supers)
        self._second_pass(lat)

        lat.meta["source_format"] = "bmad"
        if self._title is not None:
            lat.meta["bmad_title"] = self._title
        if self._beginning:
            lat.meta["twiss"] = {k: self._num(v, resolver) for k, v in self._beginning.items()
                                 if k not in ("e_tot", "p0c", "species", "particle")}
        if self._params:
            lat.meta["bmad_parameter"] = dict(self._params)
        if self._particle_start:
            lat.meta["bmad_particle_start"] = dict(self._particle_start)
        rep.raise_if(strict)
        return lat, rep

    # -- file / statement parsing ------------------------------------------
    def _parse_file(self, path: Path, text: str, depth: int) -> None:
        if depth > 32:
            raise BmadSyntaxError(f"call nesting deeper than 32 at {path}")
        for joined, lineno in logical_lines(text):
            for stmt in split_statements(joined):
                if self._statement(Statement(stmt, str(path), lineno), path, depth):
                    return                                  # end_file / return

    def _statement(self, st: Statement, path: Path, depth: int) -> bool:
        """Handle one statement; returns True when the file must stop early."""
        text = st.text.strip()
        low = text.lower()
        head = re.split(r"[\s,=]", low, maxsplit=1)[0]

        if head in ("end_file", "return"):
            return True
        if head in _IGNORED_STATEMENTS:
            return False
        if head == "call":
            self._do_call(text, path, depth)
            return False
        if head == "use":
            self._use = [t.strip().lower() for t in split_top(text[3:].lstrip(" ,")) if t.strip()]
            return False
        if head == "title":
            body = text[5:].strip().lstrip(",").strip()
            self._title = body.strip("\"'") or None
            return False
        if head in ("print", "set", "beam"):                # Bmad bookkeeping, no IR content
            return False

        m = _RE_ATTR_SET.match(text)
        if m:
            self._do_attr_set(m.group(1).lower(), m.group(2).lower(), m.group(3).strip(), st)
            return False
        m = _RE_DEFERRED_VAR.match(text)
        if m:
            self._set_variable(m.group(1), m.group(2))
            return False
        m = _RE_DEF.match(text)
        if m:
            self._do_definition(m.group(1), m.group(2).strip(), st)
            return False
        m = _RE_VAR.match(text)
        if m and not m.group(3).strip().startswith("="):
            self._set_variable(m.group(1), m.group(3))
            return False
        self._warnings.append(f"{st.file}:{st.line}: unrecognised Bmad statement {text[:70]!r}")
        return False

    def _set_variable(self, name: str, text: str) -> None:
        key = name.lower()
        if key not in self._vars:
            self._var_order.append(key)
        self._vars[key] = text.strip()

    def _do_call(self, text: str, path: Path, depth: int) -> None:
        parts = split_top(text[4:].lstrip(" ,"))
        target = None
        for p in parts:
            if "=" in p:
                key, val = p.split("=", 1)
                key = key.strip().lower()
                # Bmad accepts any prefix of FILENAME ("f", "fi", "file", ...)
                if key and "filename".startswith(key):
                    target = val.strip().strip("\"'")
            elif target is None:
                target = p.strip().strip("\"'")
        if not target:
            self._warnings.append(f"{path}: 'call' without a file name: {text[:60]!r}")
            return
        sub = (path.parent / target).resolve()
        if not sub.is_file():
            self._rep.lossy("CALL_FILE_MISSING", f"call, file = {target!r} does not exist",
                            element=None, kind=None, file=str(sub))
            self._warnings.append(f"{path}: call target {sub} not found")
            return
        self._parse_file(sub, sub.read_text(encoding="utf-8", errors="replace"), depth + 1)

    def _do_attr_set(self, name: str, attr: str, value: str, st: Statement) -> None:
        if name == "parameter":
            self._params[attr] = value
        elif name == "beginning":
            self._beginning[attr] = value
        elif name in ("particle_start", "beam_start"):
            self._particle_start[attr] = value
        elif name in self._defs:
            self._defs[name].attrs[attr] = value
            self._defs[name].flags.discard(attr)
        elif name in self._lines:
            self._line_attrs.setdefault(name, {})[attr] = value
        else:
            self._warnings.append(
                f"{st.file}:{st.line}: attribute override for unknown element {name!r}")

    def _do_definition(self, name: str, body: str, st: Statement) -> None:
        parts = split_top(body)
        if not parts:
            raise BmadSyntaxError(f"{st.file}:{st.line}: empty definition for {name!r}")
        head = parts[0]
        low = head.lower()
        if low.startswith("line") or low.startswith("list"):
            self._do_line(name, head, parts[1:], st)
            return
        type_token = low.split("=")[0].strip()
        if type_token in _CONTROLLER_TYPES:
            d = _Def(name, type_token, st.file, st.line)
            d.attrs["_body"] = body
            self._register(d)
            return
        d = _Def(name, type_token, st.file, st.line)
        for part in parts[1:]:
            self._attribute(d, part, st)
        self._register(d)

    def _do_line(self, name: str, head: str, rest: list[str], st: Statement) -> None:
        body = head.split("=", 1)[1] if "=" in head else ""
        if rest:                       # the item list was split on its own commas
            body = ",".join([body] + rest)
        body = body.strip()
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1]
        key = name.lower()
        self._lines[key] = split_top(body)
        low = head.lower()
        if "multipass" in low:
            self._rep.lossy("MULTIPASS_LINE",
                            f"line {name!r} is a multipass line; the IR expands it as repetition "
                            "and loses the shared-lord relationship",
                            element=name, kind=None)
        elif low.lstrip().startswith("list"):
            # a Bmad `list` hands out its members one per use, cycling; the IR has no
            # such construct, so it is expanded once like a line.
            self._rep.lossy("BMAD_LIST_AS_LINE",
                            f"{name!r} is a Bmad 'list' (its members are handed out one per "
                            "occurrence, cycling); the IR expands it once as an ordinary line",
                            element=name, kind=None)

    def _attribute(self, d: _Def, part: str, st: Statement) -> None:
        p = part.strip()
        if not p:
            return
        if p.startswith("{"):
            d.terms.append(p)
            return
        low = p.lower()
        if low.startswith("call::"):
            sub = (Path(d.file).parent / p[6:].strip().strip("\"'")).resolve()
            if sub.is_file():
                extra = strip_comment(sub.read_text(encoding="utf-8", errors="replace"))
                for piece in split_top(extra.replace("\n", " ")):
                    self._attribute(d, piece, st)
            else:
                self._rep.lossy("CALL_FILE_MISSING", f"call::{p[6:].strip()!r} does not exist",
                                element=d.name, kind=None)
            return
        if "=" not in p:
            d.flags.add(low)
            return
        key, val = p.split("=", 1)
        d.attrs[key.strip().lower()] = val.strip()

    def _register(self, d: _Def) -> None:
        if d.type in self._defs:                     # element template inheritance
            d.inherit(self._defs[d.type])
        if d.name not in self._defs:
            self._order.append(d.name)
        self._defs[d.name] = d

    # -- expressions --------------------------------------------------------
    def _make_resolver(self) -> LazyResolver:
        """Variables plus ``ele[attr]`` references, resolved lazily (Bmad is live)."""

        def attr_resolver(ref: str) -> float:
            name, _, attr = ref.partition("[")
            attr = attr.rstrip("]").strip().lower()
            name = name.strip().lower()
            src: dict[str, str] | None = None
            if name == "parameter":
                src = self._params
            elif name == "beginning":
                src = self._beginning
            elif name in ("particle_start", "beam_start"):
                src = self._particle_start
            elif name in self._defs:
                src = self._defs[name].attrs
            elif name in self._line_attrs:
                src = self._line_attrs[name]
            if src is None or attr not in src:
                raise ExpressionError(f"unknown Bmad reference {ref!r}")
            return evaluate(src[attr], {}, self._resolver)

        self._resolver = LazyResolver(self._vars, "infix", attr_resolver)
        return self._resolver

    def _num(self, text: str | None, resolver: LazyResolver, default: float = 0.0) -> float:
        if text is None:
            return default
        s = str(text).strip().strip("\"'")
        if not s:
            return default
        low = s.lower()
        if low in ("t", "true"):
            return 1.0
        if low in ("f", "false"):
            return 0.0
        try:
            return evaluate(s, {}, resolver)
        except ExpressionError as e:
            self._warnings.append(f"cannot evaluate Bmad expression {s!r}: {e}")
            return default

    def _variables(self, resolver: LazyResolver) -> dict[str, Variable]:
        out: dict[str, Variable] = {}
        for name in self._var_order:
            txt = self._vars[name]
            value = self._num(txt, resolver)
            expr = Expression(text=txt, deferred=not _is_literal(txt), dialect="infix") if self._keep else None
            out[name] = Variable(value=value, expression=expr)
        return out

    # -- reference particle -------------------------------------------------
    def _reference(self, resolver: LazyResolver, species: str | Species | None,
                   kinetic_energy_eV: float | None) -> ReferenceParticle:
        # Bmad's default reference species is POSITRON (measured: a deck with no
        # parameter[particle] reports REF_SPECIES = Positron).
        token = (self._params.get("particle") or self._beginning.get("particle")
                 or self._params.get("species") or "positron").strip().strip("\"'").lower()
        if species is not None:
            sp = species if isinstance(species, Species) else species_by_name(species)
        else:
            key = SPECIES_FROM_BMAD.get(token)
            if key is None:
                try:
                    sp = species_by_name(token)
                except KeyError:
                    self._warnings.append(
                        f"parameter[particle] = {token!r} is not a species lattix knows; "
                        "assuming positron (pass species= to override)")
                    sp = SPECIES["positron"]
                else:
                    key = sp.name
            if key is not None:
                sp = SPECIES[key]

        ke = kinetic_energy_eV
        if ke is None:
            e_tot = self._params.get("e_tot") or self._beginning.get("e_tot")
            p0c = self._params.get("p0c") or self._beginning.get("p0c")
            if e_tot is not None:
                ke = self._num(e_tot, resolver) - sp.mass_eV
            elif p0c is not None:
                pc = self._num(p0c, resolver)
                ke = math.hypot(pc, sp.mass_eV) - sp.mass_eV
            else:
                self._warnings.append("deck sets neither parameter[e_tot] nor parameter[p0c]; "
                                      "the reference energy defaults to 1 GeV total")
                ke = 1e9 - sp.mass_eV
        if ke <= 0:
            self._warnings.append(
                f"the deck's reference kinetic energy is not positive ({ke} eV) for species "
                f"{sp.name!r}; clamped to 1 eV so the lattice can still be walked")
            self._rep.lossy("REFERENCE_ENERGY_INVALID",
                            f"e_tot/p0c gives a non-positive kinetic energy ({ke} eV) for "
                            f"{sp.name!r}; clamped to 1 eV",
                            element=None, kind=None, kinetic_energy_eV=ke)
            ke = 1.0
        freq = None
        for d in self._defs.values():
            if "rf_frequency" in d.attrs:
                freq = self._num(d.attrs["rf_frequency"], resolver) or None
                break
        return ReferenceParticle(species=sp, kinetic_energy_eV=ke, rf_frequency_Hz=freq)

    # -- element construction ----------------------------------------------
    def _root_line(self, use: str | None) -> str:
        if use is not None:
            if use.lower() not in self._lines:
                raise KeyError(f"deck has no line {use!r}; known: {sorted(self._lines)}")
            return use.lower()
        for cand in self._use:
            if cand in self._lines:
                return cand
        if self._lines:
            name = list(self._lines)[-1]
            self._warnings.append(f"deck has no usable 'use' statement; using line {name!r}")
            return name
        raise ValueError("Bmad deck defines no line")

    def _merge_line_attrs(self, root: str) -> None:
        """``l1[e_tot] = 1e6`` / ``l1[beta_a] = 10`` act like the global statements."""
        for attr, raw in self._line_attrs.get(root, {}).items():
            # a line parameter is the branch's own value and wins over the global one
            if attr in ("e_tot", "p0c", "geometry", "particle", "species", "n_part"):
                self._params[attr] = raw
                self._params.pop("p0c" if attr == "e_tot" else "e_tot", None)
            else:
                self._beginning[attr] = raw

    def _build_elements(self, lat: Lattice, resolver: LazyResolver,
                        file: str) -> list[tuple[str, dict[str, str]]]:
        """Create one IR element per definition; returns the superimpose queue."""
        supers: list[tuple[str, dict[str, str]]] = []
        for name in self._order:
            d = self._defs[name]
            el = self._convert(d, resolver)
            tag = self._tags.get(name, {})
            el.provenance = Provenance(format="bmad", file=file, line=d.line,
                                       original_name=tag.get("name", d.src_name),
                                       original_type=(tag.get("type") if tag else d.type))
            self._apply_common(el, d, resolver)
            lat.elements[name] = el
            if "superimpose" in d.flags or self._truthy(d.attrs.get("superimpose")):
                supers.append((name, d.attrs))
        return supers

    @staticmethod
    def _truthy(text: str | None) -> bool:
        return bool(text) and str(text).strip().lower() in ("t", "true", "1", "y", "yes")

    def _convert(self, d: _Def, resolver: LazyResolver) -> Element:
        handler = getattr(self, f"_conv_{d.type}", None)
        n_before = len(self._rep.entries)
        if handler is None:
            if d.type in _CONTROLLER_TYPES:
                el: Element = Directive(name=d.name, length=0.0, format="bmad", card=d.type,
                                        role="control", args=[d.attrs.get("_body", "")])
                el.native["bmad"] = {"type": d.type, "body": d.attrs.get("_body", "")}
                self._rep.lossy("BMAD_CONTROLLER",
                                f"{d.type} {d.name!r} controls other elements; the IR keeps it as "
                                "a Directive and the controlled values as they stand",
                                element=d.name, kind="Directive", bmad_type=d.type)
                return el
            el = Marker(name=d.name, length=self._num(d.attrs.get("l"), resolver))
            el.native["bmad"] = {"type": d.type, "attrs": dict(d.attrs)}
            self._rep.dropped("UNSUPPORTED_BMAD_TYPE",
                              f"Bmad element type {d.type!r} has no IR mapping; kept as a marker",
                              element=d.name, kind="Marker", bmad_type=d.type)
            return el
        el = handler(d, resolver)
        if len(self._rep.entries) == n_before:      # one ledger entry per source element
            self._rep.exact(d.name, el.kind)
        return el

    # -- per-type converters -------------------------------------------------
    def _conv_drift(self, d, r):
        el = Drift(name=d.name, length=self._num(d.attrs.get("l"), r))
        self._expr(el, d, "l", "length")
        return el

    _conv_pipe = _conv_drift

    def _conv_marker(self, d, r):
        el = Marker(name=d.name, length=self._num(d.attrs.get("l"), r))
        self._expr(el, d, "l", "length")
        return el

    _conv_fork = _conv_marker
    _conv_photon_fork = _conv_marker
    _conv_null_ele = _conv_marker

    def _conv_monitor(self, d, r):
        el = Instrument(name=d.name, length=self._num(d.attrs.get("l"), r),
                        family=_INSTRUMENT_FAMILY.get(d.type, "INSTRUMENT"))
        self._expr(el, d, "l", "length")
        return el

    _conv_instrument = _conv_monitor
    _conv_detector = _conv_monitor

    def _conv_quadrupole(self, d, r):
        el = Quadrupole(name=d.name, length=self._num(d.attrs.get("l"), r))
        self._thick_strength(el, d, r, order=1, key="k1", field_key="b1_gradient")
        return el

    def _conv_sextupole(self, d, r):
        el = Sextupole(name=d.name, length=self._num(d.attrs.get("l"), r))
        self._thick_strength(el, d, r, order=2, key="k2", field_key="b2_gradient")
        return el

    def _conv_octupole(self, d, r):
        el = Octupole(name=d.name, length=self._num(d.attrs.get("l"), r))
        self._thick_strength(el, d, r, order=3, key="k3", field_key="b3_gradient")
        return el

    def _thick_strength(self, el: Element, d: _Def, r, *, order: int, key: str,
                        field_key: str) -> None:
        nat = el.native.setdefault("bmad", {})
        if field_key in d.attrs:
            el.multipole.Bn[order] = self._num(d.attrs[field_key], r)
            self._rep.equivalent("BMAD_FIELD_MASTER",
                                 f"{field_key} is Bmad's k·p0c/c and is charge-sign agnostic; the "
                                 "IR takes it as the lab field",
                                 element=d.name, kind=el.kind, attr=field_key)
        else:
            nat[key] = self._num(d.attrs.get(key), r)
        tilt = self._tilt(d, r, order)
        if tilt:
            el.multipole.tilt[order] = tilt
        self._expr(el, d, key, f"multipole.Bn[{order}]")
        self._expr(el, d, "l", "length")

    def _tilt(self, d: _Def, r, order: int) -> float:
        """``tilt`` on a multipole magnet; a bare flag means Bmad's π/(2(n+1))."""
        if "tilt" in d.flags:
            return math.pi / (2 * (order + 1))
        return self._num(d.attrs.get("tilt"), r)

    def _conv_multipole(self, d, r):
        el = Multipole(name=d.name, length=0.0)
        nat = el.native.setdefault("bmad", {})
        kl: dict[int, float] = {}
        ksl: dict[int, float] = {}
        tilts: dict[int, float] = {}
        for attr, raw in d.attrs.items():
            m = _RE_MULT_KL.match(attr)
            if m:
                (ksl if m.group(2) else kl)[int(m.group(1))] = self._num(raw, r)
                continue
            m = _RE_MULT_T.match(attr)
            if m:
                tilts[int(m.group(1))] = self._num(raw, r)
        for flag in d.flags:
            m = _RE_MULT_T.match(flag)
            if m:
                n = int(m.group(1))
                tilts[n] = math.pi / (2 * (n + 1))
        # a rotated normal multipole is the (b_n, a_n) pair; fold t_n into the skew part
        for n in sorted(set(kl) | set(ksl) | set(tilts)):
            a, b = ksl.get(n, 0.0), kl.get(n, 0.0)
            t = tilts.get(n, 0.0)
            if t:
                c, s = math.cos((n + 1) * t), math.sin((n + 1) * t)
                b, a = b * c + a * s, a * c - b * s
            if b:
                nat.setdefault("knl", {})[n] = b
            if a:
                nat.setdefault("ksl", {})[n] = a
        if "tilt" in d.attrs or "tilt" in d.flags:
            el.multipole.tilt[0] = math.pi / 4.0 if "tilt" in d.flags \
                else self._num(d.attrs.get("tilt"), r)
        return el

    def _conv_ab_multipole(self, d, r):
        el = Multipole(name=d.name, length=self._num(d.attrs.get("l"), r))
        nat = el.native.setdefault("bmad", {})
        for attr, raw in d.attrs.items():
            m = _RE_MULT_AB.match(attr)
            if not m:
                continue
            n = int(m.group(2))
            v = self._num(raw, r) * math.factorial(n)          # b_n = k_nl / n!  (measured)
            if v:
                nat.setdefault("ksl" if m.group(1) == "a" else "knl", {})[n] = v
        if el.length:
            self._rep.equivalent("THICK_MULTIPOLE_AS_THIN",
                                 f"{d.type} {d.name!r} has l > 0; the IR keeps the length and the "
                                 "integrated strengths (a thin kick at the centre)",
                                 element=d.name, kind="Multipole", length=el.length)
        return el

    _conv_thick_multipole = _conv_ab_multipole

    def _conv_sbend(self, d, r):
        return self._conv_bend(d, r, rect=False)

    def _conv_rbend(self, d, r):
        return self._conv_bend(d, r, rect=True)

    _conv_rf_bend = _conv_sbend

    def _conv_bend(self, d, r, *, rect: bool):
        arc_given = rect and "l_arc" in d.attrs
        length = self._num(d.attrs.get("l_arc") if arc_given else d.attrs.get("l"), r)
        angle = self._num(d.attrs.get("angle"), r)
        g = self._num(d.attrs.get("g"), r)
        if not angle and g:
            # Bmad: angle = g * L_arc; an rbend's own ``l`` is the CHORD, so
            # L_arc = 2 asin(g L_chord / 2) / g (elements.tex:796-806, measured)
            angle = (g * length if (not rect or arc_given)
                     else 2.0 * math.asin(max(-1.0, min(1.0, g * length / 2.0))))
        if rect and angle and not arc_given:
            length = length * (angle / 2.0) / math.sin(angle / 2.0)   # chord -> arc (measured)
        half = angle / 2.0 if rect else 0.0
        fint = 0.5 if "fint" in d.flags else self._num(d.attrs.get("fint"), r)
        if "fintx" in d.flags:
            fintx: float | None = 0.5
        elif "fintx" in d.attrs:
            fintx = self._num(d.attrs["fintx"], r)
        else:
            fintx = None                     # Bmad's own rule: fintx defaults to fint
        el = Bend(name=d.name, length=length)
        el.bend = BendP(
            angle=angle,
            e1=self._num(d.attrs.get("e1"), r) + half,
            e2=self._num(d.attrs.get("e2"), r) + half,
            edge_int1=fint,
            edge_int2=fintx,
            hgap=self._num(d.attrs.get("hgap"), r),
            tilt_ref=self._num(d.attrs.get("ref_tilt"), r),
            rect=rect,
        )
        nat = el.native.setdefault("bmad", {})
        for key, order in (("k1", 1), ("k2", 2)):
            if key in d.attrs:
                nat[key] = self._num(d.attrs[key], r)
                self._expr(el, d, key, f"multipole.Bn[{order}]")
        for k in ("fringe_type", "exact_multipole", "l_chord", "rho"):
            if k in d.attrs:
                nat[k] = d.attrs[k]
        dropped = {k: self._num(d.attrs[k], r) for k in ("dg", "b_field", "hgapx")
                   if k in d.attrs}
        if dropped:
            nat.update(dropped)
            self._rep.lossy("BEND_ATTR_DROPPED",
                            f"bend attributes not modelled by the IR: {sorted(dropped)}",
                            element=d.name, kind="Bend", **dropped)
        self._expr(el, d, "angle", "bend.angle")
        self._expr(el, d, "e1", "bend.e1")
        self._expr(el, d, "e2", "bend.e2")
        self._expr(el, d, "l", "length")
        return el

    def _conv_solenoid(self, d, r):
        el = Solenoid(name=d.name, length=self._num(d.attrs.get("l"), r))
        nat = el.native.setdefault("bmad", {})
        if "bs_field" in d.attrs:
            el.solenoid.Bsol_T = self._num(d.attrs["bs_field"], r)
            self._rep.equivalent("BMAD_FIELD_MASTER",
                                 "bs_field is Bmad's ks·p0c/c and is charge-sign agnostic; the IR "
                                 "takes it as the lab field",
                                 element=d.name, kind="Solenoid", attr="bs_field")
        else:
            nat["ks"] = self._num(d.attrs.get("ks"), r)
        self._expr(el, d, "ks", "solenoid.Bsol_T")
        self._expr(el, d, "l", "length")
        return el

    def _conv_sol_quad(self, d, r):
        el = self._conv_solenoid(d, r)
        k1 = self._num(d.attrs.get("k1"), r)
        if k1:
            el.native.setdefault("bmad", {})["k1"] = k1
            self._rep.lossy("SOLQUAD_K1_DROPPED",
                            f"sol_quad {d.name!r} carries k1 = {k1!r}; the IR solenoid keeps only "
                            "the solenoid field (the gradient stays in native['bmad'])",
                            element=d.name, kind="Solenoid", k1=k1)
        return el

    def _conv_lcavity(self, d, r):
        el = RFCavity(name=d.name, length=self._num(d.attrs.get("l"), r))
        rf = el.rf
        rf.frequency_Hz = self._num(d.attrs.get("rf_frequency"), r) or None
        phi0 = self._num(d.attrs.get("phi0"), r) + self._num(d.attrs.get("phi0_err"), r)
        rf.phase_rad = phase_from_bmad_phi0(phi0)
        if "gradient" in d.attrs and "voltage" not in d.attrs:
            rf.gradient_V_per_m = self._num(d.attrs["gradient"], r)
            rf.voltage_V = rf.gradient_V_per_m * el.length
        else:
            rf.voltage_V = self._num(d.attrs.get("voltage"), r)
            if "gradient" in d.attrs:
                rf.gradient_V_per_m = self._num(d.attrs["gradient"], r)
        if "l_active" in d.attrs:
            rf.L_active_m = self._num(d.attrs["l_active"], r)
        if "n_cell" in d.attrs:
            rf.n_cell = int(self._num(d.attrs["n_cell"], r))
        ctype = str(d.attrs.get("cavity_type", "")).strip().lower()
        if ctype.startswith("traveling"):
            rf.cavity_type = "TRAVELING_WAVE"
        nat = el.native.setdefault("bmad", {})
        for key in ("phi0_multipass", "coupler_strength", "coupler_at", "coupler_phase",
                    "gradient_err", "voltage_err", "autoscale_phase", "autoscale_amplitude",
                    "cavity_type", "n_rf_steps", "longitudinal_mode", "harmon"):
            if key in d.attrs:
                nat[key] = d.attrs[key]
        self._expr(el, d, "voltage", "rf.voltage_V")
        self._expr(el, d, "gradient", "rf.gradient_V_per_m")
        self._expr(el, d, "phi0", "rf.phase_rad")
        self._expr(el, d, "rf_frequency", "rf.frequency_Hz")
        self._expr(el, d, "l", "length")
        return el

    def _conv_rfcavity(self, d, r):
        el = self._conv_lcavity(d, r)
        # Bmad's rfcavity is a ring cavity: phi0 = 0 is the zero crossing, not the crest
        # (measured on 20260828.0: the reference particle gains no energy for any phi0).
        phi0 = self._num(d.attrs.get("phi0"), r) + self._num(d.attrs.get("phi0_multipass"), r)
        el.rf.phase_rad = phase_from_bmad_phi0(0.25 - phi0)
        el.rf.dE_ref_eV = 0.0
        el.native.setdefault("bmad", {})["key"] = "rfcavity"
        self._rep.equivalent("RING_RFCAVITY",
                             "Bmad 'rfcavity' uses -sin(2π(φ_t - phi0)) where 'lcavity' uses "
                             "+cos(2π(φ_t + phi0)), so the IR synchronous phase is the reflection "
                             "2π(0.25 - phi0) (bmad/doc/attributes.tex:2551); the reference energy "
                             "does not change (measured on 20260828.0), so dE_ref = 0",
                             element=d.name, kind="RFCavity", phi0=phi0)
        return el

    def _conv_hkicker(self, d, r):
        el = Kicker(name=d.name, length=self._num(d.attrs.get("l"), r))
        el.native.setdefault("bmad", {})["hkick"] = self._num(d.attrs.get("kick"), r)
        self._expr(el, d, "kick", "hkick")
        return el

    def _conv_vkicker(self, d, r):
        el = Kicker(name=d.name, length=self._num(d.attrs.get("l"), r))
        el.native.setdefault("bmad", {})["vkick"] = self._num(d.attrs.get("kick"), r)
        self._expr(el, d, "kick", "vkick")
        return el

    def _conv_kicker(self, d, r):
        el = Kicker(name=d.name, length=self._num(d.attrs.get("l"), r))
        nat = el.native.setdefault("bmad", {})
        nat["hkick"] = self._num(d.attrs.get("hkick"), r)
        nat["vkick"] = self._num(d.attrs.get("vkick"), r)
        for key, ir in (("bl_hkick", "hkick"), ("bl_vkick", "vkick")):
            if key in d.attrs:                       # integrated field: already a lab quantity
                nat.pop(ir, None)
                nat[key] = self._num(d.attrs[key], r)
        self._expr(el, d, "hkick", "hkick")
        self._expr(el, d, "vkick", "vkick")
        return el

    _conv_ac_kicker = _conv_kicker

    def _conv_elseparator(self, d, r):
        el = self._conv_kicker(d, r)
        el.electric = True
        self._rep.equivalent("ELSEPARATOR_AS_KICKER",
                             "Bmad 'elseparator' modelled as an electric IR Kicker (its gap and "
                             "voltage stay in native['bmad'])",
                             element=d.name, kind="Kicker")
        for key in ("gap", "voltage", "e_field"):
            if key in d.attrs:
                el.native.setdefault("bmad", {})[key] = self._num(d.attrs[key], r)
        return el

    def _conv_rcollimator(self, d, r):
        el = Collimator(name=d.name, length=self._num(d.attrs.get("l"), r))
        el.native.setdefault("bmad", {})["shape"] = "RECTANGULAR"
        return el

    def _conv_ecollimator(self, d, r):
        el = Collimator(name=d.name, length=self._num(d.attrs.get("l"), r))
        el.native.setdefault("bmad", {})["shape"] = "ELLIPTICAL"
        return el

    _conv_collimator = _conv_rcollimator
    _conv_mask = _conv_rcollimator

    def _conv_foil(self, d, r):
        el = Foil(name=d.name, length=self._num(d.attrs.get("l"), r))
        el.material = str(d.attrs.get("material_type", "C")).strip().strip("\"'")
        thickness = self._num(d.attrs.get("thickness"), r)
        density = d.attrs.get("density")
        area = self._num(d.attrs.get("area_density"), r)
        if area:
            el.thickness_kg_per_m2 = area
        elif thickness and density is not None:
            el.thickness_kg_per_m2 = thickness * self._num(split_top(density.strip("()"))[0], r)
        nat = el.native.setdefault("bmad", {})
        for key in ("thickness", "density", "radiation_length", "area_density", "scatter_method",
                    "dthickness_dx", "num_steps"):
            if key in d.attrs:
                nat[key] = d.attrs[key]
        if thickness and not el.thickness_kg_per_m2:
            self._rep.equivalent("FOIL_THICKNESS_GEOMETRIC",
                                 "foil thickness is a length in Bmad; the IR's areal density needs "
                                 "a density, which the deck does not give",
                                 element=d.name, kind="Foil", thickness=thickness)
        return el

    def _conv_taylor(self, d, r):
        el = Taylor(name=d.name, length=self._num(d.attrs.get("l"), r))
        higher: list[str] = []
        for term in d.terms:
            m = _RE_TERM.match(term.strip())
            if not m:
                continue
            out, coef, idx = m.group(1).strip().lower(), m.group(2).strip(), m.group(3).strip()
            if not out.isdigit():                    # {s1: …} spin terms
                higher.append(term.strip())
                continue
            i = int(out) - 1
            value = self._num(coef, r)
            if idx == "":
                el.offset[i] = value
            elif len(idx) == 1 and idx.isdigit():
                el.matrix[i][int(idx) - 1] = value
            else:
                higher.append(term.strip())
        if higher:
            el.native.setdefault("bmad", {})["higher_order_terms"] = higher
            self._rep.lossy("TAYLOR_ORDER_TRUNCATED",
                            f"{len(higher)} Taylor terms above first order (and spin terms) are "
                            "kept in native['bmad'] but not modelled",
                            element=d.name, kind="Taylor", n_terms=len(higher))
        return el

    def _conv_patch(self, d, r):
        el = Patch(name=d.name, length=self._num(d.attrs.get("l"), r))
        el.x_offset = self._num(d.attrs.get("x_offset"), r)
        el.y_offset = self._num(d.attrs.get("y_offset"), r)
        el.z_offset = self._num(d.attrs.get("z_offset"), r)
        el.x_rot = self._num(d.attrs.get("x_pitch"), r)
        el.y_rot = self._num(d.attrs.get("y_pitch"), r)
        el.tilt = self._num(d.attrs.get("tilt"), r)
        nat = el.native.setdefault("bmad", {})
        dropped = {}
        for key in ("t_offset", "e_tot_offset", "e_tot_set", "p0c_set", "flexible", "ref_coords"):
            if key in d.attrs:
                nat[key] = d.attrs[key]
                if key in ("e_tot_offset", "e_tot_set", "p0c_set", "t_offset"):
                    dropped[key] = d.attrs[key]
        if dropped:
            self._rep.lossy("PATCH_ATTR_DROPPED",
                            f"patch attributes not modelled by the IR Patch: {sorted(dropped)}",
                            element=d.name, kind="Patch", **dropped)
        return el

    def _conv_em_field(self, d, r):
        el = FieldMap(name=d.name, length=self._num(d.attrs.get("l"), r))
        nat = el.native.setdefault("bmad", {})
        files: list[str] = []
        for key in ("grid_field", "cartesian_map", "cylindrical_map", "gen_grad_map"):
            if key in d.attrs:
                nat[key] = d.attrs[key]
                for m in re.finditer(r'file\s*=\s*["\']?([^"\',}]+)', d.attrs[key], re.IGNORECASE):
                    files.append(m.group(1).strip())
        el.files = files
        for key in ("field_calc", "tracking_method", "ds_step", "num_steps"):
            if key in d.attrs:
                nat[key] = d.attrs[key]
        if "rf_frequency" in d.attrs:
            el.rf.frequency_Hz = self._num(d.attrs["rf_frequency"], r) or None
        if "phi0" in d.attrs:
            el.rf.phase_rad = phase_from_bmad_phi0(self._num(d.attrs["phi0"], r))
        self._rep.equivalent("BMAD_GRID_FIELD",
                             "Bmad field map kept verbatim in native['bmad']; its reference energy "
                             "gain is unknown to the IR until the map is integrated (Phase 3)",
                             element=d.name, kind="FieldMap", files=files)
        return el

    _conv_wiggler = _conv_em_field
    _conv_undulator = _conv_em_field

    # -- shared attributes (aperture, misalignment) --------------------------
    def _apply_common(self, el: Element, d: _Def, r) -> None:
        self._apply_aperture(el, d, r)
        self._apply_shift(el, d, r)
        for key in ("type", "alias", "descrip", "num_steps", "ds_step", "tracking_method",
                    "mat6_calc_method", "field_calc", "integrator_order", "spin_tracking_method"):
            if key in d.attrs:
                el.native.setdefault("bmad", {})[key] = d.attrs[key]
        if "num_steps" in d.attrs:
            el.tracking["n_steps"] = int(self._num(d.attrs["num_steps"], r))

    def _apply_aperture(self, el: Element, d: _Def, r) -> None:
        a = d.attrs
        x1 = x2 = y1 = y2 = None
        if "aperture" in a:
            v = self._num(a["aperture"], r)
            x1 = x2 = y1 = y2 = v
        if "x_limit" in a:
            x1 = x2 = self._num(a["x_limit"], r)
        if "y_limit" in a:
            y1 = y2 = self._num(a["y_limit"], r)
        for key, slot in (("x1_limit", "x1"), ("x2_limit", "x2"),
                          ("y1_limit", "y1"), ("y2_limit", "y2")):
            if key in a:
                v = self._num(a[key], r)
                if slot == "x1":
                    x1 = v
                elif slot == "x2":
                    x2 = v
                elif slot == "y1":
                    y1 = v
                else:
                    y2 = v
        if not any(v for v in (x1, x2, y1, y2) if v):
            return
        default = "elliptical" if d.type == "ecollimator" else "rectangular"
        kind = str(a.get("aperture_type", default)).strip().lower()
        shape = "ELLIPTICAL" if kind.startswith("ellip") else "RECTANGULAR"
        at = str(a.get("aperture_at", "exit_end")).strip().lower()
        location = {"entrance_end": "ENTRANCE", "exit_end": "EXIT", "both_ends": "BOTH_ENDS",
                    "continuous": "CONTINUOUS"}.get(at, "EXIT")
        el.aperture = ApertureP(shape=shape, aperture_at=location,
                                x_limits=(-(x1 or 0.0), (x2 or 0.0)) if (x1 or x2) else None,
                                y_limits=(-(y1 or 0.0), (y2 or 0.0)) if (y1 or y2) else None)
        if kind not in ("rectangular", "elliptical", ""):
            el.native.setdefault("bmad", {})["aperture_type"] = kind
            self._rep.lossy("APERTURE_TYPE_UNSUPPORTED",
                            f"aperture_type {kind!r} approximated by its bounding rectangle",
                            element=d.name, kind=el.kind, aperture_type=kind)

    def _apply_shift(self, el: Element, d: _Def, r) -> None:
        a = d.attrs
        tilt_is_physics = isinstance(el, (Quadrupole, Sextupole, Octupole, Multipole, Patch))
        tilt = 0.0
        if isinstance(el, Bend):
            tilt = self._num(a.get("roll"), r)
        elif not tilt_is_physics:
            tilt = self._num(a.get("tilt"), r)
            if "tilt" in d.flags:
                tilt = math.pi / 4.0
        shift = BodyShiftP(x_offset=self._num(a.get("x_offset"), r),
                           y_offset=self._num(a.get("y_offset"), r),
                           z_offset=self._num(a.get("z_offset"), r),
                           x_rot=self._num(a.get("x_pitch"), r),
                           y_rot=self._num(a.get("y_pitch"), r),
                           tilt=tilt)
        if isinstance(el, Patch):                    # its offsets ARE the element
            return
        if not shift.is_zero():
            el.shift = shift

    def _expr(self, el: Element, d: _Def, attr: str, path: str) -> None:
        """Record the deck's source text for *attr* (a normalized Bmad quantity)."""
        if not self._keep or attr not in d.attrs:
            return
        text = d.attrs[attr].strip()
        if not text or _is_number(text):
            return
        el.expressions[path] = Expression(text=text, deferred=True, dialect="infix")
        el.native.setdefault("bmad", {})[f"{attr}_expr"] = text

    # -- lines ---------------------------------------------------------------
    def _build_lines(self, lat: Lattice, root: str) -> None:
        for name in self._lines:
            lat.lines[name] = Line(name=name, items=[])
        for name, items in list(self._lines.items()):
            lat.lines[name].items = [self._line_item(lat, it) for it in items]
        lat.use = root
        lat.name = root

    def _line_item(self, lat: Lattice, text: str) -> LineItem:
        s = text.strip()
        reverse = False
        repeat = 1
        while s.startswith("-"):
            reverse = not reverse
            s = s[1:].strip()
        m = re.match(r"^(\d+)\s*\*\s*(.*)$", s)
        if m:
            repeat = int(m.group(1))
            s = m.group(2).strip()
        while s.startswith("-"):
            reverse = not reverse
            s = s[1:].strip()
        if s.startswith("(") and s.endswith(")"):
            self._anon += 1
            name = f"__anon_line_{self._anon}"
            self._lines[name] = split_top(s[1:-1])
            lat.lines[name] = Line(name=name,
                                   items=[self._line_item(lat, it) for it in self._lines[name]])
            s = name
        return LineItem(ref=s.lower(), repeat=repeat, reverse=reverse)

    # -- superimpose ---------------------------------------------------------
    def _resolve_superimpose(self, lat: Lattice, supers: list[tuple[str, dict[str, str]]]) -> None:
        """Insert superimposed elements into a flattened root line.

        Bmad slices whatever the element overlaps; the IR has no slice concept, so
        the reader only resolves the cases that stay expressible: a zero-length
        element goes in at its s, a thick one splits a :class:`Drift`.  Anything
        else is left out of the line and recorded ``LOSSY:SUPERIMPOSE_UNSUPPORTED``
        with its ``ref``/``offset`` in ``native['bmad']``.
        """
        root = lat.use or lat.name
        placed = lat.flatten(root)
        seq: list[Element] = [p.element for p in placed]
        by_name: dict[str, tuple[float, float]] = {}
        for p in placed:
            by_name.setdefault(p.element.name, (p.s_in, p.s_out))

        for name, attrs in supers:
            el = lat.elements[name]
            nat = el.native.setdefault("bmad", {})
            nat["superimpose"] = {k: v for k, v in attrs.items()
                                  if k in ("ref", "offset", "ref_origin", "ele_origin",
                                           "create_jumbo_slave", "wrap_superimpose")}
            ref_name = str(attrs.get("ref", "")).strip().lower()
            offset = self._num(attrs.get("offset"), self._resolver)
            if ref_name and ref_name not in by_name:
                self._rep.lossy("SUPERIMPOSE_UNSUPPORTED",
                                f"superimpose ref {ref_name!r} is not in the used line",
                                element=name, kind=el.kind, ref=ref_name, offset=offset)
                continue
            r_in, r_out = by_name.get(ref_name, (0.0, 0.0))
            origin = str(attrs.get("ref_origin", "center")).strip().lower()
            base = {"beginning": r_in, "end": r_out}.get(origin, (r_in + r_out) / 2.0)
            eo = str(attrs.get("ele_origin", "center")).strip().lower()
            centre = base + offset
            s0 = {"beginning": centre, "end": centre - el.length}.get(eo, centre - el.length / 2.0)
            ok = self._insert_at(seq, el, s0)
            if ok:
                self._rep.equivalent("SUPERIMPOSE_RESOLVED",
                                     f"superimposed {name!r} inserted at s = {s0:.9g} m (the root "
                                     "line is flattened)",
                                     element=name, kind=el.kind, s=s0, ref=ref_name)
            else:
                self._rep.lossy("SUPERIMPOSE_UNSUPPORTED",
                                f"superimposed {name!r} overlaps a non-drift element; the IR "
                                "cannot slice it, so it is not placed in the line",
                                element=name, kind=el.kind, ref=ref_name, offset=offset, s=s0)

        kept = dict(lat.elements)
        flat = Line(name=root, items=[])
        lat.elements = {}
        for el in seq:
            flat.items.append(LineItem(ref=lat.add_element(el)))
        for nm, el in kept.items():          # definitions that never reached the line survive
            lat.elements.setdefault(nm, el)
        lat.lines = {root: flat}
        lat.use = root

    @staticmethod
    def _insert_at(seq: list[Element], el: Element, s0: float) -> bool:
        """Place *el* so that it starts at *s0*, splitting a drift when needed."""
        tol = 1e-9
        s = 0.0
        for i, cur in enumerate(seq):
            if abs(s - s0) <= tol:
                seq.insert(i, el)
                return True
            end = s + cur.length
            if s0 < end - tol:
                if not isinstance(cur, Drift) or s0 + el.length > end + tol:
                    return False
                head = cur.model_copy(deep=True)
                head.name = f"{cur.name}_1"
                head.length = s0 - s
                tail = cur.model_copy(deep=True)
                tail.name = f"{cur.name}_2"
                tail.length = end - (s0 + el.length)
                new = ([head] if head.length > tol else []) + [el] + \
                      ([tail] if tail.length > tol else [])
                seq[i:i + 1] = new
                return True
            s = end
        if abs(s - s0) <= tol:
            seq.append(el)
            return True
        return False

    # -- second pass: normalized strengths -> lab fields --------------------
    def _second_pass(self, lat: Lattice) -> None:
        """Convert every ``native['bmad']`` normalized value with the LOCAL rigidity.

        Bmad's reference energy follows ``lcavity``, so ``Bρ`` is a function of s;
        the walk gives each occurrence its own entrance reference and the *first*
        occurrence of a definition sets its lab field (a definition reused at two
        energies is recorded ``EQUIVALENT:MULTI_RIGIDITY_DEFINITION``).
        """
        seen: dict[int, float] = {}
        for p in propagate(lat, warnings=self._warnings):
            el = p.element
            brho = (p.ref_in or lat.reference).brho_signed
            key = id(el)
            if key in seen:
                if abs(seen[key] - brho) > 1e-12 * max(1.0, abs(seen[key])):
                    self._rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                                         "one definition appears at two reference energies; the "
                                         "first occurrence's rigidity sets its lab fields",
                                         element=el.name, kind=el.kind,
                                         brho_first=seen[key], brho_here=brho)
                continue
            seen[key] = brho
            self._denormalize(el, brho)
        for el in lat.elements.values():          # definitions outside the used line
            if id(el) not in seen:
                self._denormalize(el, lat.reference.brho_signed)

    @staticmethod
    def _denormalize(el: Element, brho: float) -> None:
        nat = el.native.get("bmad")
        if not nat:
            return
        if isinstance(el, (Quadrupole, Sextupole, Octupole)):
            order = {"Quadrupole": 1, "Sextupole": 2, "Octupole": 3}[el.kind]
            key = f"k{order}"
            if key in nat:
                el.multipole.Bn[order] = nat.pop(key) * brho
        elif isinstance(el, Bend):
            for key, order in (("k1", 1), ("k2", 2)):
                if key in nat:
                    el.multipole.Bn[order] = nat.pop(key) * brho
        elif isinstance(el, Multipole):
            for key, table in (("knl", el.multipole.BnL), ("ksl", el.multipole.BsL)):
                for order, v in nat.pop(key, {}).items():
                    table[int(order)] = v * brho
        elif isinstance(el, Solenoid):
            if "ks" in nat:
                el.solenoid.Bsol_T = nat.pop("ks") * brho
        elif isinstance(el, Kicker):
            for key, attr in (("hkick", "hkick"), ("vkick", "vkick")):
                if key in nat:
                    setattr(el, attr, nat.pop(key))
            for key, attr in (("bl_hkick", "hkick"), ("bl_vkick", "vkick")):
                if key in nat:
                    setattr(el, attr, nat.pop(key) / brho)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def read(path: str | Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
