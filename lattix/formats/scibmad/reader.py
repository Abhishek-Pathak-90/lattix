"""SciBmad (Beamlines.jl) reader: the Julia subset :mod:`lattix.formats.scibmad.writer` and Bmad's
own ``bmad_to_scibmad`` produce.

Understood: ``using …`` lines (ignored), ``# lattix: …`` tags, ``function name(v, q, p=nothing) …
end`` transport maps (linear terms; anything of higher order is noted and dropped), the
``@elements begin … end`` block of ``name = Kind(key = value, …)`` definitions (multi-line, any
Julia arithmetic in the values), ``name = [a, b, …]`` vectors and ``name = Beamline([…]; pc_ref = …,
species_ref = Species("…"))`` (also ``E_ref``, ``p_over_q_ref``; items may be splatted vectors
``v...``, ``reverse(v)...`` or ``repeat(v, n)...``).  Everything else is a warning, never a crash.

Strengths come back through the beamline's rigidity (SciBmad keeps one reference momentum), then
the ``energy_mode`` tag is undone with the same walk the writer used (:mod:`lattix.ir.energy_mode`);
the measured gain sign (:data:`lattix.formats.scibmad.writer.GAIN_SIGN`) is inverted on the voltage.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Drift,
    Element,
    Foil,
    Freq,
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
    SolenoidP,
    Taylor,
)
from lattix.ir.energy_mode import ENERGY_MODES, restore_energy_mode, undo_phase_slip
from lattix.ir.expr import ExpressionError, evaluate
from lattix.ir.lattice import Lattice, Line, LineItem
from lattix.ir.reference import ReferenceParticle, Species
from lattix.ir.reference import species as species_by_name
from lattix.ir.reference_tag import parse_reference_tag
from lattix.ir.units import wrap_rad

from .writer import GAIN_SIGN, SPECIES_NAMES

_TAG_LINE = re.compile(r"^\s*#\s*lattix:\s*(?P<body>.*?)\s*$")
_TAG_KV = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_ENERGY_MODE = re.compile(r"^\s*#\s*lattix:\s*energy_mode=(\w+)", re.M)
_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_!]*)\s*=\s*(.*)$", re.S)
_CALL = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\((.*)\)\s*$", re.S)
_FUNCTION = re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_TERM = re.compile(r"([+-]?)\s*([0-9.]+(?:[eE][+-]?\d+)?)\s*(?:\*\s*v\[(\d)\])?")
_SPECIES_BACK = {v: k for k, v in SPECIES_NAMES.items()}
_SPECIES_BACK.update({"H-": "h-", "anti-proton": "antiproton", "antiproton": "antiproton"})
#: reference keywords Beamlines.jl accepts on a Beamline call and, as the leading element of a
#: line, on a Marker (HELIX's SciBmad examples and Bmad's own converter write the Marker form)
_REF_KEYS = ("species_ref", "E_ref", "pc_ref", "p_over_q_ref")
_MULT = re.compile(r"^(Kn|Ks|Bn|Bs)(\d+)(L?)$")
_TILT = re.compile(r"^tilt(\d+)$")


class ScibmadSyntaxError(ValueError):
    pass


# ---------------------------------------------------------------------------
# tokenizing helpers

def strip_comment(line: str) -> str:
    """Drop a ``#`` comment (outside strings)."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out)


def split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split on *sep* outside brackets and strings."""
    parts, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _balanced(text: str) -> bool:
    depth, quote = 0, None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ('"', "'"):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return depth <= 0


def _statements(lines: list[str]) -> list[tuple[int, str]]:
    """Join physical lines into statements with balanced brackets; ``(line number, text)``."""
    out: list[tuple[int, str]] = []
    buf, start = "", 0
    for i, raw in enumerate(lines, 1):
        code = strip_comment(raw).rstrip()
        if not code.strip():
            continue
        if not buf:
            start = i
        buf = (buf + "\n" + code) if buf else code
        if _balanced(buf):
            out.append((start, buf))
            buf = ""
    if buf:
        out.append((start, buf))
    return out


# ---------------------------------------------------------------------------
# value parsing

class _Value:
    """A parsed keyword value: ``number`` (float), ``text`` (string/enum/identifier) or ``call``."""

    __slots__ = ("kind", "value", "raw")

    def __init__(self, kind: str, value: Any, raw: str) -> None:
        self.kind, self.value, self.raw = kind, value, raw


def parse_value(raw: str, variables: dict[str, float] | None = None) -> _Value:
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return _Value("text", s[1:-1], s)
    if s in ("true", "false"):
        return _Value("bool", s == "true", s)
    m = _CALL.match(s)
    if m:
        return _Value("call", (m.group(1), m.group(2)), s)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$", s):     # ApertureShape.Rectangular
        return _Value("enum", s.split(".", 1)[1], s)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_!]*$", s) and s not in ("pi", "e"):
        return _Value("ident", s, s)
    try:
        return _Value("number", float(evaluate(s.replace("π", "pi"), variables or {})), s)
    except (ExpressionError, ValueError, TypeError):
        return _Value("text", s, s)


def parse_kwargs(body: str, variables: dict[str, float] | None = None) -> tuple[list[str], dict[str, _Value]]:
    """``(positional, {key: value})`` of a call body (``;`` separates keywords in Julia too)."""
    positional: list[str] = []
    kw: dict[str, _Value] = {}
    for part in split_top_level(body.replace(";", ",")):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.*)$", part, re.S)
        if m:
            kw[m.group(1)] = parse_value(m.group(2), variables)
        else:
            positional.append(part)
    return positional, kw


# ---------------------------------------------------------------------------
# transport-map functions

def parse_map_function(lines: list[str]) -> tuple[list[list[float]], list[float], bool]:
    """Linear part of a ``function f(v, q, p=nothing) … end`` body: ``(matrix, offset, truncated)``.

    Understands ``vN = c*v[k] + … + c0`` (lattix) and ``v_outN = …`` spread over several lines
    (Bmad's converter); products of coordinates (``v[1]^2``, ``v[1]*v[4]``) mark the map truncated."""
    matrix = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
    offset = [0.0] * 6
    truncated = False
    joined = " ".join(strip_comment(ln).strip() for ln in lines[1:])
    joined = joined.replace("return", " return ")
    for m in re.finditer(r"v(?:_out)?(\d)\s*=\s*(.*?)(?=\s(?:v(?:_out)?\d|q(?:_out)?\d)\s*=|\sreturn\s|\send\b|$)",
                         joined):
        i = int(m.group(1)) - 1
        expr = m.group(2).strip()
        if not (0 <= i < 6):
            continue
        if re.search(r"v\[\d\]\s*(\^|\*\s*v\[)", expr) or "q[" in expr:
            truncated = True
        row = [0.0] * 6
        const = 0.0
        # strip nonlinear products so the linear terms parse cleanly
        expr_lin = re.sub(r"[+-]?\s*[0-9.]+(?:[eE][+-]?\d+)?\s*\*\s*(v\[\d\]\s*(\^\d|\*\s*[vq]\[\d\]))", "", expr)
        for t in _TERM.finditer(expr_lin):
            sign = -1.0 if t.group(1) == "-" else 1.0
            c = sign * float(t.group(2))
            if t.group(3):
                row[int(t.group(3)) - 1] += c
            else:
                const += c
        matrix[i] = row
        offset[i] = const
    return matrix, offset, truncated


# ---------------------------------------------------------------------------
# the reader

class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "scibmad"

    def read(self, path: Path, *, strict: bool = False, use: str | None = None,
             species: str | Species | None = None, kinetic_energy_eV: float | None = None,
             keep_expressions: bool = False) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        rep = FidelityReport(source_format="scibmad", source_file=str(path))
        warnings: list[str] = []
        lines = text.splitlines()

        tags = self._tags(lines)
        functions = self._functions(lines)
        elements_raw, vectors, beamlines = self._statements(lines, rep, warnings)

        # -- reference --------------------------------------------------------
        root_name = use or (beamlines[-1][0] if beamlines else None)
        root_kw: dict[str, _Value] = {}
        for name, _items, kw in beamlines:
            if name == root_name:
                root_kw = kw
        ref_kw = dict(root_kw)
        for key, val in self._marker_reference(root_name, beamlines, vectors, elements_raw).items():
            ref_kw.setdefault(key, val)                 # the Beamline's own keywords win
        ref = self._reference(ref_kw, species, kinetic_energy_eV, text, rep, warnings)
        brho = ref.brho_signed

        # -- elements ----------------------------------------------------------
        lat = Lattice(name=root_name or path.stem, reference=ref, warnings=warnings)
        jl_to_ir: dict[str, str] = {}
        for line_no, jname, kind, body in elements_raw:
            tag = tags.get(jname, {})
            el = self._element(jname, kind, body, tag, functions, brho, ref, rep, warnings, line_no)
            # the original type travels in the tag only: re-writing a deck must not grow a
            # type="SBend" tag (write -> read -> write is a fixed point)
            el.provenance = Provenance(format="scibmad", file=str(path), line=line_no,
                                       original_name=tag.get("name", jname),
                                       original_type=tag.get("type"))
            registered = lat.add_element(el)
            jl_to_ir[jname] = registered

        # -- lines -------------------------------------------------------------
        for vname, items in vectors:
            lat.lines[vname] = Line(name=vname, items=self._items(items, jl_to_ir, vectors, rep, warnings))
        for name, items, _kw in beamlines:
            lat.lines[name] = Line(name=name, items=self._items(items, jl_to_ir, vectors, rep, warnings))
        if root_name is None:
            rep.lossy("NO_BEAMLINE", "the file defines no Beamline; every element is placed once in "
                      "definition order", element=None, kind=None)
            root_name = path.stem
            lat.lines[root_name] = Line(name=root_name,
                                        items=[LineItem(ref=jl_to_ir[j]) for _l, j, _k, _b in elements_raw])
        lat.use = root_name
        lat.meta["source_format"] = "scibmad"
        lat.meta["source_file"] = str(path)

        mode_m = _ENERGY_MODE.search(text)
        mode = mode_m.group(1).lower() if mode_m else None
        if mode in ENERGY_MODES:
            if mode == "delta":
                undo_phase_slip(lat, rep)
            restore_energy_mode(lat, rep, mode)
        rep.raise_if(strict)
        return lat, rep

    # -- scanning -----------------------------------------------------------
    @staticmethod
    def _tags(lines: list[str]) -> dict[str, dict[str, str]]:
        """``# lattix: …`` comment lines attach to the next definition (``name = …``)."""
        out: dict[str, dict[str, str]] = {}
        pending: dict[str, str] | None = None
        for raw in lines:
            m = _TAG_LINE.match(raw)
            if m:
                body = m.group("body")
                if body.startswith(("energy_mode", "reference ", "directive")):
                    continue
                kv = dict(_TAG_KV.findall(body))
                if kv:
                    pending = kv
                continue
            code = strip_comment(raw).strip()
            if not code:
                continue
            a = _ASSIGN.match(code)
            if a and pending is not None:
                out[a.group(1)] = pending
            pending = None
        return out

    @staticmethod
    def _functions(lines: list[str]) -> dict[str, tuple[list[list[float]], list[float], bool]]:
        out: dict[str, tuple[list[list[float]], list[float], bool]] = {}
        i = 0
        while i < len(lines):
            m = _FUNCTION.match(lines[i])
            if not m:
                i += 1
                continue
            j = i + 1
            depth = 0
            while j < len(lines):
                code = strip_comment(lines[j]).strip()
                if re.match(r"^(function|if|for|while|let|begin|try|do)\b", code) or code.endswith(" do"):
                    depth += 1
                elif code == "end":
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            out[m.group(1)] = parse_map_function(lines[i:j + 1])
            i = j + 1
        return out

    def _statements(self, lines: list[str], rep: FidelityReport, warnings: list[str]):
        """``(elements, vectors, beamlines)`` from the file's statements."""
        elements: list[tuple[int, str, str, str]] = []
        vectors: list[tuple[str, list[str]]] = []
        beamlines: list[tuple[str, list[str], dict[str, _Value]]] = []
        in_block = False
        in_function = 0
        for line_no, stmt in _statements(lines):
            first = stmt.split("\n", 1)[0].strip()
            if _FUNCTION.match(first):
                in_function += 1
                continue
            if in_function:
                if first == "end":
                    in_function -= 1
                continue
            if first.startswith("@elements"):
                in_block = True
                continue
            if first == "end" and in_block:
                in_block = False
                continue
            if first.startswith(("using ", "import ", "include(", "@")):
                continue
            a = _ASSIGN.match(stmt)
            if not a:
                warnings.append(f"line {line_no}: statement ignored: {first[:60]!r}")
                continue
            name, rhs = a.group(1), a.group(2).strip()
            call = _CALL.match(rhs)
            if call and call.group(1) == "Beamline":
                items, kw = self._beamline_call(call.group(2))
                beamlines.append((name, items, kw))
            elif rhs.startswith("[") and rhs.endswith("]"):
                vectors.append((name, split_top_level(rhs[1:-1])))
            elif call and in_block:
                elements.append((line_no, name, call.group(1), call.group(2)))
            elif call:
                elements.append((line_no, name, call.group(1), call.group(2)))   # tolerate definitions outside
            else:
                warnings.append(f"line {line_no}: assignment ignored: {first[:60]!r}")
        return elements, vectors, beamlines

    @staticmethod
    def _beamline_call(body: str) -> tuple[list[str], dict[str, _Value]]:
        parts = split_top_level(body.replace(";", ","))
        items: list[str] = []
        kw: dict[str, _Value] = {}
        for part in parts:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*(.*)$", part, re.S)
            if m:
                kw[m.group(1)] = parse_value(m.group(2))
            elif part.startswith("[") and part.endswith("]"):
                items += split_top_level(part[1:-1])
            else:
                items.append(part)             # a vector variable
        return items, kw

    # -- reference ------------------------------------------------------------
    @staticmethod
    def _marker_reference(root_name, beamlines, vectors, elements_raw) -> dict[str, _Value]:
        """The reference keywords of a ``Marker`` placed first in the root line.

        ``Beamline([...]; species_ref = …, E_ref = …)`` is one carrier; the other is a leading
        ``lat_begin = Marker(species_ref = …, E_ref = …)``, which is what HELIX's SciBmad examples
        and Bmad's own converter emit.  Anything else, a splat of a vector included, resolves to
        its own first entry; an unresolvable head simply carries no reference.
        """
        items = next((it for name, it, _kw in beamlines if name == root_name), None)
        if items is None:
            items = [jname for _l, jname, _k, _b in elements_raw][:1]
        head, seen = (items[0].strip() if items else ""), set()
        vecs = dict(vectors)
        while head:
            head = head.removesuffix("...").strip()
            if head in vecs and head not in seen and vecs[head]:
                seen.add(head)
                head = vecs[head][0].strip()
                continue
            break
        for _line_no, jname, kind, body in elements_raw:
            if jname == head and kind == "Marker":
                _pos, kw = parse_kwargs(body)
                return {k: v for k, v in kw.items() if k in _REF_KEYS}
        return {}

    def _reference(self, kw: dict[str, _Value], species_opt, kinetic_opt, text: str,
                   rep: FidelityReport, warnings: list[str]) -> ReferenceParticle:
        tag = parse_reference_tag(text)
        sp: Species | None = None
        if species_opt is not None:
            sp = species_by_name(species_opt) if isinstance(species_opt, str) else species_opt
        elif "species_ref" in kw:
            sp = self._species(kw["species_ref"], rep, warnings)
        elif tag is not None:
            sp = tag.species
        if sp is None:
            sp = species_by_name("proton")
            rep.lossy("SPECIES_ASSUMED", "no species_ref on the Beamline, its leading Marker or a reference "
                                  "tag; proton assumed",
                      element=None, kind=None)
        f = tag.rf_frequency_Hz if tag is not None else None
        if kinetic_opt is not None:
            return ReferenceParticle(species=sp, kinetic_energy_eV=float(kinetic_opt), rf_frequency_Hz=f)
        if "pc_ref" in kw and kw["pc_ref"].kind == "number":
            return ReferenceParticle.from_momentum(sp, kw["pc_ref"].value, rf_frequency_Hz=f)
        if "E_ref" in kw and kw["E_ref"].kind == "number":
            return ReferenceParticle.from_total_energy(sp, kw["E_ref"].value, rf_frequency_Hz=f)
        if "p_over_q_ref" in kw and kw["p_over_q_ref"].kind == "number":
            return ReferenceParticle.from_brho(sp, abs(kw["p_over_q_ref"].value), rf_frequency_Hz=f)
        if tag is not None:
            rep.equivalent("REFERENCE_FROM_TAG", "reference energy taken from the '# lattix: reference' tag",
                           element=None, kind=None)
            return ReferenceParticle(species=sp, kinetic_energy_eV=tag.kinetic_energy_eV, rf_frequency_Hz=f)
        rep.lossy("ENERGY_ASSUMED", "no pc_ref/E_ref on the Beamline or its leading Marker; "
                  "1 GeV kinetic assumed "
                  "(normalized strengths are meaningless without it)", element=None, kind=None)
        return ReferenceParticle(species=sp, kinetic_energy_eV=1e9, rf_frequency_Hz=f)

    @staticmethod
    def _species(v: _Value, rep: FidelityReport, warnings: list[str]) -> Species | None:
        if v.kind == "call" and v.value[0] == "Species":
            args = split_top_level(v.value[1])
            if not args:
                return None
            name = args[0].strip().strip('"')
            if len(args) >= 3:
                try:
                    return Species(name=name, charge=int(round(float(evaluate(args[1], {})))),
                                   mass_eV=float(evaluate(args[2], {})))
                except (ExpressionError, ValueError):
                    warnings.append(f"cannot parse the Species constructor {v.raw!r}")
            key = _SPECIES_BACK.get(name, name.lower())
            try:
                return species_by_name(key)
            except (KeyError, ValueError):
                rep.lossy("UNKNOWN_SPECIES", f"species {name!r} is not in lattix's table; proton assumed",
                          element=None, kind=None)
                return None
        warnings.append(f"unexpected species_ref value {v.raw!r}")
        return None

    # -- lines -----------------------------------------------------------------
    @staticmethod
    def _items(items: list[str], jl_to_ir: dict[str, str], vectors: list[tuple[str, list[str]]],
               rep: FidelityReport, warnings: list[str]) -> list[LineItem]:
        vnames = {v for v, _ in vectors}
        out: list[LineItem] = []
        for raw in items:
            s = raw.strip()
            repeat, reverse = 1, False
            if s.endswith("..."):
                s = s[:-3].strip()
            m = re.match(r"^reverse\((.*)\)$", s)
            if m:
                reverse, s = True, m.group(1).strip()
            m = re.match(r"^repeat\((.*),\s*(\d+)\)$", s)
            if m:
                s, repeat = m.group(1).strip(), int(m.group(2))
            if s.startswith("[") and s.endswith("]"):
                inner = split_top_level(s[1:-1])
                if len(inner) == 1:
                    s = inner[0]
                else:
                    for sub in inner:
                        out.append(LineItem(ref=jl_to_ir.get(sub, sub), repeat=repeat, reverse=reverse))
                    continue
            if s in vnames:
                out.append(LineItem(ref=s, repeat=repeat, reverse=reverse))
            elif s in jl_to_ir:
                out.append(LineItem(ref=jl_to_ir[s], repeat=repeat, reverse=reverse))
            else:
                warnings.append(f"beamline item {raw!r} is neither an element nor a vector; skipped")
        return out

    # -- one element ------------------------------------------------------------
    def _element(self, jname: str, kind: str, body: str, tag: dict[str, str], functions: dict, brho: float,
                 ref: ReferenceParticle, rep: FidelityReport, warnings: list[str], line_no: int) -> Element:
        _pos, kw = parse_kwargs(body)
        ir_name = jname                     # the original name lives in the provenance (as every reader does)
        length = kw["L"].value if "L" in kw and kw["L"].kind == "number" else 0.0
        tag_kind = tag.get("kind")
        el: Element

        if tag_kind == "ReferenceChange":
            fields = {f: float(tag[f]) for f in ("dE_ref_eV", "energy_eV", "dtime_s", "dphase_rad") if f in tag}
            el = ReferenceChange(name=ir_name, **fields)
            rep.equivalent("REFCHANGE_FROM_TAG", "reference change restored from the marker's lattix tag "
                           "(SciBmad itself keeps one reference energy)", element=ir_name, kind="ReferenceChange")
        elif tag_kind == "Freq":
            el = Freq(name=ir_name, frequency_Hz=float(tag.get("frequency_Hz", 0.0)))
            rep.exact(ir_name, "Freq")
        elif tag_kind == "Foil":
            el = Foil(name=ir_name, length=length, material=tag.get("material", "C"),
                      thickness_kg_per_m2=float(tag.get("thickness_kg_per_m2", 0.0)))
            if "dE_ref_eV" in tag:
                el.dE_ref_eV = float(tag["dE_ref_eV"])
            rep.equivalent("FOIL_FROM_TAG", "foil restored from its lattix tag (SciBmad has no foil physics)",
                           element=ir_name, kind="Foil")
        elif tag_kind == "Instrument":
            el = Instrument(name=ir_name, length=length, family=tag.get("family", "MONITOR"))
            rep.exact(ir_name, "Instrument")
        elif tag_kind == "Collimator":
            el = Collimator(name=ir_name, length=length)
            rep.exact(ir_name, "Collimator")
        elif kind == "Drift":
            el = Drift(name=ir_name, length=length)
            rep.exact(ir_name, "Drift")
        elif kind == "Marker":
            el = Marker(name=ir_name)
            rep.exact(ir_name, "Marker")
        elif kind in ("Quadrupole", "Sextupole", "Octupole", "Multipole", "Solenoid"):
            cls = {"Quadrupole": Quadrupole, "Sextupole": Sextupole, "Octupole": Octupole,
                   "Multipole": Multipole, "Solenoid": Solenoid}[kind]
            el = cls(name=ir_name, length=length)
            self._multipoles(el, kw, brho)
            if kind == "Solenoid":
                el.solenoid = SolenoidP(Bsol_T=self._field(kw, "Ksol", "Bsol", brho))
            rep.exact(ir_name, el.kind)
        elif kind == "SBend":
            el = self._bend(ir_name, length, kw, tag, brho, rep)
        elif kind in ("RFCavity", "CrabCavity"):
            el = self._cavity(ir_name, length, kw, tag, rep)
            if kind == "CrabCavity":
                rep.lossy("CRAB_CAVITY_AS_RF", "a crab cavity is read as an ordinary RF cavity",
                          element=ir_name, kind="RFCavity")
        elif kind in ("Kicker", "HKicker", "VKicker"):
            el = Kicker(name=ir_name, length=length)
            # a kick is an angle: the normalized integrated strength directly, a field integral over Bρ
            kn0l = self._field(kw, "Kn0L", "Bn0L", brho) / brho
            ks0l = self._field(kw, "Ks0L", "Bs0L", brho) / brho
            if "Kn0" in kw or "Bn0" in kw:
                kn0l += self._field(kw, "Kn0", "Bn0", brho) / brho * length
            if "Ks0" in kw or "Bs0" in kw:
                ks0l += self._field(kw, "Ks0", "Bs0", brho) / brho * length
            el.hkick, el.vkick = -kn0l, ks0l
            if tag.get("electric") == "true":
                el.electric = True
            rep.exact(ir_name, "Kicker")
        elif kind == "Patch":
            el = Patch(name=ir_name, length=length)
            for key, ir_key in (("dx", "x_offset"), ("dy", "y_offset"), ("dz", "z_offset"),
                                ("dx_rot", "x_rot"), ("dy_rot", "y_rot"), ("dz_rot", "tilt")):
                if key in kw and kw[key].kind == "number":
                    setattr(el, ir_key, kw[key].value)
            if "dt" in kw and kw["dt"].kind == "number":
                el.t_offset_s = kw["dt"].value
            if "e_tot_offset_eV" in tag:
                el.e_tot_offset_eV = float(tag["e_tot_offset_eV"])
            rep.exact(ir_name, "Patch")
        elif kind == "LineElement":
            el = Taylor(name=ir_name, length=length, basis="bmad")
            fn = kw.get("transport_map")
            if fn is not None and fn.kind == "ident" and fn.value in functions:
                matrix, offset, truncated = functions[fn.value]
                el.matrix, el.offset = matrix, offset
                if truncated:
                    rep.lossy("TAYLOR_ORDER_TRUNCATED", "only the linear part of the transport map was read",
                              element=ir_name, kind="Taylor")
                else:
                    rep.exact(ir_name, "Taylor")
            else:
                rep.lossy("TRANSPORT_MAP_UNKNOWN", "LineElement without a parsable transport_map function; "
                          "identity map", element=ir_name, kind="Taylor")
            if tag.get("basis") in ("madx", "elegant", "bmad", "xtrack", "tracewin", "flame", "impactx", "common"):
                el.basis = tag["basis"]
        else:
            el = Marker(name=ir_name) if not length else Drift(name=ir_name, length=length)
            rep.dropped("UNSUPPORTED_SCIBMAD_KIND", f"SciBmad {kind!r} has no IR mapping; kept as a "
                        f"{el.kind.lower()} of the same length", element=ir_name, kind=el.kind, scibmad_kind=kind)
        el.native.setdefault("scibmad", {})["kind"] = kind
        self._aperture(el, kw, tag)
        self._alignment(el, kw)
        if "meta" in tag:
            el.meta["scibmad_tag"] = tag["meta"]
        return el

    # -- attribute groups ---------------------------------------------------------
    @staticmethod
    def _field(kw: dict[str, _Value], k_key: str, b_key: str, brho: float) -> float:
        """A lab field from a normalized (``K…``) or field (``B…``) keyword."""
        if k_key in kw and kw[k_key].kind == "number":
            return kw[k_key].value * brho
        if b_key in kw and kw[b_key].kind == "number":
            return kw[b_key].value
        return 0.0

    @staticmethod
    def _multipoles(el: Element, kw: dict[str, _Value], brho: float) -> None:
        mp = getattr(el, "multipole", None)
        if mp is None:
            return
        for key, v in kw.items():
            if v.kind != "number":
                continue
            m = _MULT.match(key)
            if m:
                fam, order, integrated = m.group(1), int(m.group(2)), bool(m.group(3))
                value = v.value * (brho if fam[0] == "K" else 1.0)
                table = {("n", False): mp.Bn, ("s", False): mp.Bs, ("n", True): mp.BnL, ("s", True): mp.BsL}[
                    (fam[1], integrated)]
                if fam == "Ksol" or key in ("Ksol", "Bsol"):
                    continue
                table[order] = table.get(order, 0.0) + value
                continue
            t = _TILT.match(key)
            if t:
                mp.tilt[int(t.group(1))] = v.value

    def _bend(self, name: str, length: float, kw: dict[str, _Value], tag: dict[str, str], brho: float,
              rep: FidelityReport) -> Bend:
        g = kw["g_ref"].value if "g_ref" in kw and kw["g_ref"].kind == "number" else 0.0
        b = BendP(angle=g * length)
        for key in ("e1", "e2", "tilt_ref"):
            if key in kw and kw[key].kind == "number":
                setattr(b, key, kw[key].value)
        hgap = float(tag["hgap"]) if "hgap" in tag else None
        if "edge1_int" in kw and kw["edge1_int"].kind == "number":
            product = kw["edge1_int"].value
            product2 = kw["edge2_int"].value if "edge2_int" in kw and kw["edge2_int"].kind == "number" else product
            if hgap:
                b.hgap = hgap
                b.edge_int1 = product / hgap
                b.edge_int2 = product2 / hgap
            else:
                b.hgap = 1.0
                b.edge_int1 = product
                b.edge_int2 = product2
                if product or product2:
                    rep.equivalent("FRINGE_AS_PRODUCT", "SciBmad stores fint·hgap; hgap taken as 1 m and fint "
                                   "as the product (the fringe correction depends on the product only)",
                                   element=name, kind="Bend")
        elif "fint" in tag:
            b.edge_int1 = float(tag["fint"])
            if "fintx" in tag:
                b.edge_int2 = float(tag["fintx"])
            if hgap is not None:
                b.hgap = hgap
        if tag.get("rect") == "true":
            b.rect = True
        if "fringe_k2" in tag:
            b.fringe_k2 = float(tag["fringe_k2"])
        el = Bend(name=name, length=length, bend=b)
        self._multipoles(el, kw, brho)
        k0 = el.multipole.Bn.pop(0, None)
        if k0 is not None:
            k0 = k0 / brho
            if abs(k0 - g) > 1e-12 * max(1.0, abs(g)):
                el.native.setdefault("scibmad", {})["k0"] = k0
                rep.equivalent("BEND_K0_NE_G", f"Kn0={k0!r} differs from g_ref={g!r}; the IR keeps the geometric "
                               "angle (an energy-mode ratio explains it when the tag says so)",
                               element=name, kind="Bend", k0=k0, g_ref=g)
        else:
            rep.exact(name, "Bend")
        if not any(e.element == name for e in rep.entries):
            rep.exact(name, "Bend")
        return el

    @staticmethod
    def _cavity(name: str, length: float, kw: dict[str, _Value], tag: dict[str, str],
                rep: FidelityReport) -> RFCavity:
        rf = RFP()
        if "voltage" in kw and kw["voltage"].kind == "number":
            rf.voltage_V = kw["voltage"].value / GAIN_SIGN
        if "phi0" in kw and kw["phi0"].kind == "number":
            rf.phase_rad = wrap_rad(kw["phi0"].value)
        if "rf_frequency" in kw and kw["rf_frequency"].kind == "number":
            rf.frequency_Hz = kw["rf_frequency"].value
        if "harmon" in kw and kw["harmon"].kind == "number":
            rf.harmon = kw["harmon"].value
        if "traveling_wave" in kw and kw["traveling_wave"].kind == "bool" and kw["traveling_wave"].value:
            rf.cavity_type = "TRAVELING_WAVE"
        if "n_cell" in tag:
            rf.n_cell = int(float(tag["n_cell"]))
        if "L_active_m" in tag:
            rf.L_active_m = float(tag["L_active_m"])
        if "dE_ref_eV" in tag:
            rf.dE_ref_eV = float(tag["dE_ref_eV"])
        if tag.get("phase_is_sync") == "false":
            rf.phase_is_sync = False
        zp = kw.get("zero_phase")
        if zp is not None and zp.kind == "enum" and zp.value != "Accelerating":
            # phi0 = 0 sits at the zero crossing there (measured): shift to the IR's crest convention
            rf.phase_rad = wrap_rad(rf.phase_rad + math.pi / 2.0)
            rep.equivalent("ZERO_PHASE_SHIFTED", f"zero_phase = {zp.value}: phi0 is measured from the zero "
                           "crossing; +90 deg applied for the IR's crest convention", element=name, kind="RFCavity")
        el = RFCavity(name=name, length=length, rf=rf)
        if not any(e.element == name for e in rep.entries):
            rep.exact(name, "RFCavity")
        return el

    @staticmethod
    def _aperture(el: Element, kw: dict[str, _Value], tag: dict[str, str]) -> None:
        keys = ("x1_limit", "x2_limit", "y1_limit", "y2_limit")
        vals = {k: kw[k].value for k in keys if k in kw and kw[k].kind == "number"}
        if not vals:
            return
        shape = "ELLIPTICAL"
        s = kw.get("aperture_shape")
        if s is not None and s.kind == "enum" and s.value.lower().startswith("rect"):
            shape = "RECTANGULAR"
        ap = ApertureP(shape=shape)
        if "x1_limit" in vals or "x2_limit" in vals:
            ap.x_limits = (vals.get("x1_limit", -math.inf), vals.get("x2_limit", math.inf))
        if "y1_limit" in vals or "y2_limit" in vals:
            ap.y_limits = (vals.get("y1_limit", -math.inf), vals.get("y2_limit", math.inf))
        if tag.get("aperture_at") in ("ENTRANCE", "EXIT", "BOTH_ENDS", "CONTINUOUS"):
            ap.aperture_at = tag["aperture_at"]
        el.aperture = ap

    @staticmethod
    def _alignment(el: Element, kw: dict[str, _Value]) -> None:
        vals = {k: kw[k].value for k in ("x_offset", "y_offset", "z_offset", "x_rot", "y_rot", "tilt")
                if k in kw and kw[k].kind == "number" and kw[k].value}
        if vals and el.kind not in ("Patch",):
            el.shift = BodyShiftP(**vals)


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
