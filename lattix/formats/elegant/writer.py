"""Elegant ``.lte`` writer (PLAN §6 task 2.1): table driven, never silent.

Layout: a header comment, the IR variables as ``% <rpn> sto NAME`` stores, one
definition per element, then the IR's ``Line``s with nesting, ``3*NAME`` repeats
and ``-NAME`` reflections preserved, root line last, and an explicit
``USE, <root>`` (elegant would otherwise take the last line defined).

Conventions, all **measured** against ``elegant 2026.3.0`` on 2026-09-03:

* lengths m, angles rad, ``VOLT`` V, ``FREQ`` Hz, ``PHASE`` deg — no unit change
  from the IR except the phase convention;
* ``PHASE`` is charge-signed: crest is ``+90°`` for negative species and ``−90°``
  for positive ones (:func:`lattix.ir.rf.elegant_phase_deg`).  A 1 MV cavity at
  ``PHASE=-120`` gains a proton ``+866025.4 eV`` and an H⁻ ``−866025.4 eV``, so
  the species is part of the meaning of the number: every ``RFCA`` carries an
  ``EQUIVALENT:ELEGANT_PHASE_FOR_SPECIES`` entry naming it;
* ``CHANGE_P0=1`` is written on every cavity so elegant's reference momentum
  follows the gain (its default is 0);
* ``FINT`` defaults to **0.5** in elegant, so the writer emits ``FINT``
  explicitly whenever the IR value differs — an IR fringe integral of 0 must not
  silently become 0.5;
* ``RBEN``'s ``L`` is the **chord**: the writer emits
  ``L_chord = L_arc·sin(θ/2)/(θ/2)`` and pole faces with ``θ/2`` subtracted;
* normalized strengths (``K1``, ``K2``, ``K3``, ``KNL``, ``KS``) use the
  **signed** rigidity at each element's entrance from
  :func:`lattix.ir.walk.propagate`;
* names are upper-cased and sanitized to ``[A-Za-z][A-Za-z0-9_.:$]*``; a renamed
  element carries a ``! lattix: name="…"`` tag the reader parses back.

An element that came from an ``.lte`` keeps its own elegant type (so ``QUAD``
stays ``QUAD`` rather than becoming ``KQUAD``) and its unmapped attributes are
re-emitted verbatim from ``native["elegant"]``, which makes elegant → IR →
elegant a fixed point.  Anything else uses the ``RULES`` target below.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.elegant.naming import NameMap, is_valid, name_tag
from lattix.formats.elegant.reader import DEFAULT_FINT, evaluate_value
from lattix.ir.elements import (
    ALL_KINDS,
    ApertureP,
    Directive,
    Element,
    FieldMap,
    Freq,
    Superposition,
)
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import ReferenceParticle
from lattix.ir.rf import elegant_phase_deg
from lattix.ir.walk import energy_gain_eV, propagate

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_EXPR_TOL = 1e-12

#: alternative elegant spellings the writer will keep when the source used them.
_KEEP_TYPE: dict[str, frozenset[str]] = {
    "Drift": frozenset({"DRIF", "DRIFT", "EDRIFT", "CSRDRIFT", "CSRDRIF", "LSCDRIFT", "LSCDRIF"}),
    "Quadrupole": frozenset({"QUAD", "QUADRUPOLE", "KQUAD"}),
    "Sextupole": frozenset({"SEXT", "SEXTUPOLE", "KSEXT"}),
    "Octupole": frozenset({"OCTU", "OCTUPOLE", "KOCT"}),
    "Multipole": frozenset({"MULT", "MULTIPOLE"}),
    "Solenoid": frozenset({"SOLE", "SOLENOID"}),
    "RFCavity": frozenset({"RFCA", "RFCW"}),
    "Marker": frozenset({"MARK", "MARKER"}),
    "Taylor": frozenset({"EMATRIX"}),
}
#: sector-bend spellings (an IR ``rect`` bend always goes out as ``RBEN``).
_KEEP_BEND = frozenset({"SBEN", "SBEND", "CSBEND", "CSRCSBEND", "CSRCSBEN", "NIBEND", "CCBEND"})
#: only ``CSBEND`` accepts ``FINT1``/``FINT2``; the others carry a single ``FINT``.
_SPLIT_FINT = frozenset({"CSBEND", "CSRCSBEND", "CSRCSBEN"})
#: IR ``Instrument.family`` -> elegant diagnostic type.
_FAMILY_TYPE = {"BPM": "MONI", "MONITOR": "MONI", "HMON": "HMON", "VMON": "VMON",
                "WATCH": "WATCH"}
#: an element counts as accelerating above 1 µeV of reference gain.
_ACCEL_TOL_eV = 1e-6

#: which misalignment attributes each elegant type actually accepts.  Measured on
#: elegant 2026.3.0 by feeding a bogus parameter and reading its "valid parameters are"
#: list — DRIF/LSCDRIFT/KICKER/WATCH/MAXAMP accept none at all, RFCA and the collimators
#: take DX/DY only, and only the K-family and CSBEND carry pitch/yaw.
_ALIGN_CAPS: dict[str, frozenset[str]] = {
    "DRIF": frozenset(), "DRIFT": frozenset(), "EDRIFT": frozenset(),
    "LSCDRIFT": frozenset(), "LSCDRIF": frozenset(),
    "CSRDRIFT": frozenset({"DZ"}), "CSRDRIF": frozenset({"DZ"}),
    "QUAD": frozenset({"DX", "DY", "DZ", "PITCH", "YAW"}),
    "QUADRUPOLE": frozenset({"DX", "DY", "DZ", "PITCH", "YAW"}),
    "KQUAD": frozenset({"DX", "DY", "DZ", "PITCH", "YAW"}),
    "SEXT": frozenset({"DX", "DY", "DZ"}), "SEXTUPOLE": frozenset({"DX", "DY", "DZ"}),
    "KSEXT": frozenset({"DX", "DY", "DZ", "PITCH", "YAW"}),
    "OCTU": frozenset({"DX", "DY", "DZ"}), "OCTUPOLE": frozenset({"DX", "DY", "DZ"}),
    "KOCT": frozenset({"DX", "DY", "DZ", "PITCH", "YAW"}),
    "MULT": frozenset({"DX", "DY", "DZ"}), "MULTIPOLE": frozenset({"DX", "DY", "DZ"}),
    "SOLE": frozenset({"DX", "DY", "DZ"}), "SOLENOID": frozenset({"DX", "DY", "DZ"}),
    "RFCA": frozenset({"DX", "DY"}), "RFCW": frozenset({"DX", "DY"}),
    "KICKER": frozenset(), "KICK": frozenset(), "HKICK": frozenset(), "HKIC": frozenset(),
    "VKICK": frozenset(), "VKIC": frozenset(),
    "EHKICK": frozenset({"DX", "DY", "DZ"}), "EVKICK": frozenset({"DX", "DY", "DZ"}),
    "EKICKER": frozenset({"DX", "DY", "DZ"}),
    "ECOL": frozenset({"DX", "DY"}), "RCOL": frozenset({"DX", "DY"}),
    "MAXAMP": frozenset(), "WATCH": frozenset(),
    "MARK": frozenset({"DX", "DY"}), "MARKER": frozenset({"DX", "DY"}),
    "MONI": frozenset({"DX", "DY", "DZ"}), "HMON": frozenset({"DX", "DY", "DZ"}),
    "VMON": frozenset({"DX", "DY", "DZ"}),
    "EMATRIX": frozenset({"DX", "DY", "DZ", "PITCH", "YAW", "TILT"}),
    # bends: every spelling takes DX/DY/DZ and ETILT; only CSBEND adds EPITCH/EYAW
    "SBEN": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "SBEND": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "RBEN": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "RBEND": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "CSRCSBEND": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "CSRCSBEN": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "NIBEND": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "CCBEND": frozenset({"DX", "DY", "DZ", "ETILT"}),
    "CSBEND": frozenset({"DX", "DY", "DZ", "ETILT", "EPITCH", "EYAW"}),
}


def _num(x: float) -> str:
    s = f"{float(x):.15g}"
    return "0" if s in ("-0", "-0.0") else s


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str                 # the elegant construct this kind becomes
    cls: str = "EXACT"          # default fidelity class
    code: str = "OK"
    message: str = ""


@dataclass
class _Item:
    """One occurrence to emit: an element, where it sits, and its local rigidity."""

    element: Element
    s_in: float
    s_out: float
    brho: float
    dE: float


class Writer:
    """``Writer().write(lattice, path)`` -> :class:`FidelityReport`."""

    format = "elegant"

    RULES: dict[str, Rule] = {
        "Drift": Rule("DRIF"),
        "Quadrupole": Rule("KQUAD"),
        "Sextupole": Rule("KSEXT"),
        "Octupole": Rule("KOCT"),
        "Multipole": Rule("MULT"),
        "Bend": Rule("CSBEND/RBEN"),
        "Solenoid": Rule("SOLE"),
        "RFCavity": Rule("RFCA"),
        "FieldMap": Rule("RFCA/SOLE/KQUAD/DRIF", "LOSSY", "FM_TO_DRIFT",
                         "field map degraded per its integrated summary: RF → RFCA "
                         "(FM_TO_CAVITY), static solenoid/quadrupole → hard edge with drift "
                         "padding (FM_SOL_HARDEDGE / FM_QUAD_HARDEDGE), otherwise a drift"),
        "NCells": Rule("DRIF", "LOSSY", "NCELLS_TO_DRIFT",
                       "NCELLS cell train replaced by a drift of the same length"),
        "RFQCell": Rule("DRIF", "LOSSY", "RFQ_TO_DRIFT",
                        "RFQ cell replaced by a drift of the same length"),
        "Kicker": Rule("KICKER/EHKICK/EVKICK"),
        "Collimator": Rule("RCOL/ECOL"),
        "Marker": Rule("MARK"),
        "Instrument": Rule("MONI/HMON/VMON/WATCH/MARK"),
        "Foil": Rule("MARK", "LOSSY", "FOIL_TO_MARKER",
                     "elegant has no stripping foil; written as a marker"),
        "Taylor": Rule("EMATRIX"),
        "Patch": Rule("MARK", "LOSSY", "PATCH_DROPPED",
                      "elegant has no patch element; written as a marker"),
        "ReferenceChange": Rule("MARK", "LOSSY", "REFCHANGE_DROPPED",
                                "elegant cannot change the reference energy outside an RF "
                                "element; written as a marker"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each elegant cavity's FREQ attribute"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                              "overlapping fields written as consecutive elements"),
    }

    #: Directive roles elegant can honour as a plain comment without losing anything.
    COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              line_name: str | None = None, use_expressions: bool = True) -> FidelityReport:
        rep = FidelityReport(target_format="elegant", target_file=str(path))
        self._lattice = lattice
        self._values = {k: v.value for k, v in lattice.variables.items()}
        self._use_expressions = use_expressions
        self._charge = lattice.reference.species.charge
        #: element name -> the elegant names it was written as (a MULT splits by order)
        self._expanded: dict[str, list[str]] = {}

        placed = propagate(lattice)
        items = self._items(lattice, placed, rep)

        names = NameMap()
        root = lattice.use or lattice.name
        root_name = names.reserve(line_name or (root if root in lattice.lines else None)
                                  or lattice.name or "LATTIX_LINE")
        line_names: dict[str, str] = {}
        if root in lattice.lines:
            line_names[root] = root_name
            for ln in lattice.lines:
                if ln != root:
                    line_names[ln] = names.reserve(ln)

        defs: dict[int, tuple[str, Element, float]] = {}
        for it in items:
            key = id(it.element)
            if key in defs:
                brho0 = defs[key][2]
                if abs(it.brho - brho0) > 1e-12 * max(1.0, abs(brho0)):
                    rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                                   "one definition is used at two reference energies; the first "
                                   "occurrence's rigidity sets its normalized strengths",
                                   element=it.element.name, kind=it.element.kind,
                                   brho_first=brho0, brho_here=it.brho)
                continue
            defs[key] = (names.assign(it.element.name, it.element), it.element, it.brho)

        # elegant keeps definitions that no beam line uses; write them too (with the
        # rigidity at the lattice start) so elegant -> IR -> elegant loses nothing.
        start_brho = lattice.reference.brho_signed
        unused = [el for el in lattice.elements.values() if id(el) not in defs]
        for el in unused:
            defs[id(el)] = (names.assign(el.name, el), el, start_brho)
        if unused:
            rep.equivalent("DEFINITION_NOT_IN_LINE",
                           f"{len(unused)} element definition(s) are not referenced by the root "
                           "beam line; they are written with the rigidity at the lattice start",
                           element=None, kind=None, elements=[e.name for e in unused])

        body: list[str] = [f"! lattix {__version__} from {lattice.meta.get('source_format', 'IR')}",
                           f"! reference: {lattice.reference.species.name}, "
                           f"{_num(lattice.reference.kinetic_energy_eV)} eV kinetic"]
        variables = self._variable_lines(lattice, rep)
        if variables:
            body.append("")
            body.append("! --- variables")
            body.extend(variables)

        body.append("")
        body.append("! --- elements")
        seen: set[int] = set()
        for it in items:
            key = id(it.element)
            if key in seen:
                continue
            seen.add(key)
            name, el, brho = defs[key]
            for text in self._definition(name, el, brho, rep):
                body.append(text)
        for el in unused:
            name, _, brho = defs[id(el)]
            body.extend(self._definition(name, el, brho, rep))

        body.append("")
        body.append("! --- beam lines")
        body.extend(self._line_block(root_name, line_names, lattice, defs, rep))
        body.append("")
        body.append(f"USE, {root_name}")
        Path(path).write_text("\n".join(body).rstrip("\n") + "\n", encoding="utf-8")
        rep.raise_if(strict)
        return rep

    # -- flattening ---------------------------------------------------------
    @staticmethod
    def _items(lattice: Lattice, placed: list[Placed], rep: FidelityReport) -> list[_Item]:
        out: list[_Item] = []
        for p in placed:
            el = p.element
            ref: ReferenceParticle = p.ref_in or lattice.reference
            if isinstance(el, Superposition):
                rep.lossy("SUPERPOSITION_FLATTENED",
                          "overlapping fields written as consecutive elegant elements",
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

    # -- variables ----------------------------------------------------------
    def _variable_lines(self, lattice: Lattice, rep: FidelityReport) -> list[str]:
        if not self._use_expressions or not lattice.variables:
            return []
        keep = {}
        for name, var in lattice.variables.items():
            if is_valid(name):
                keep[name] = var
            else:
                rep.equivalent("VARIABLE_DROPPED",
                               f"variable {name!r} is not a writable elegant identifier; its value "
                               "is folded into the numbers",
                               element=None, kind=None, variable=name)
        lines = []
        for name in self._topo(keep):
            var = keep[name]
            expr = var.expression
            if expr is not None and expr.dialect == "rpn" and expr.text.strip():
                lines.append(f"% {expr.text.strip()} sto {name.upper()}")
            else:
                if expr is not None and expr.text.strip():
                    rep.equivalent("EXPRESSION_TO_RPN",
                                   f"variable {name!r} has a non-RPN expression "
                                   f"({expr.text.strip()!r}); its value is written instead",
                                   element=None, kind=None, variable=name)
                lines.append(f"% {_num(var.value)} sto {name.upper()}")
        return lines

    @staticmethod
    def _topo(variables: dict) -> list[str]:
        """Definition order with dependencies first (stable otherwise)."""
        deps = {name: {t for t in _ID.findall(var.expression.text if var.expression else "")
                       if t in variables and t != name}
                for name, var in variables.items()}
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

    # -- element definitions ------------------------------------------------
    @staticmethod
    def _native(el: Element) -> dict:
        return el.native.get("elegant") or {}

    def _source_type(self, el: Element) -> str | None:
        t = self._native(el).get("type")
        return str(t).upper() if t else None

    def _attr(self, el: Element, key: str, value: float, rep: FidelityReport) -> str:
        """``KEY=<source text>`` when the deck's own text still gives *value*, else the number."""
        if self._use_expressions:
            raw = self._native(el).get("attrs", {}).get(key)
            if raw is not None:
                got = evaluate_value(raw, self._values)
                if got is not None and abs(got - value) <= _EXPR_TOL * max(1.0, abs(value)):
                    return f"{key}={raw}"
        return f"{key}={_num(value)}"

    def _passthrough(self, el: Element, etype: str, written: set[str]) -> list[str]:
        """Native attributes the reader did not interpret, re-emitted verbatim."""
        nat = self._native(el)
        if not nat or str(nat.get("type", "")).upper() != etype.upper():
            return []
        consumed = {str(k).upper() for k in nat.get("consumed", ())}
        return [f"{k}={v}" for k, v in nat.get("attrs", {}).items()
                if k.upper() not in consumed and k.upper() not in written]

    def _definition(self, name: str, el: Element, brho: float, rep: FidelityReport) -> list[str]:
        rule = self.RULES.get(el.kind)
        if rule is None:                                    # pragma: no cover - RULES is total
            raise KeyError(f"elegant writer has no rule for kind {el.kind!r}")
        if isinstance(el, Freq):
            self._expanded[el.name] = []
            rep.exact(el.name, "Freq", message=rule.message)
            return []
        if isinstance(el, Directive):
            return self._directive(name, el, rule, rep)

        emitted = getattr(self, f"_def_{el.kind.lower()}")(el, brho, rep)
        out: list[str] = []
        original = (el.provenance.original_name if el.provenance else None) or el.name
        self._expanded[el.name] = [name if i == 0 else f"{name}__{i + 1}"
                                   for i in range(len(emitted))]
        for i, (etype, attrs) in enumerate(emitted):
            attrs = list(attrs)
            written = {a.split("=", 1)[0].upper() for a in attrs}
            if len(emitted) == 1:
                attrs += self._passthrough(el, etype, written)
            nm = name if i == 0 else f"{name}__{i + 1}"
            tag = ("   " + name_tag(original, el.provenance.original_type if el.provenance else None)
                   if nm != original else "")
            joined = ", ".join(attrs)
            out.append(f"{nm}: {etype}" + (f", {joined}" if joined else "") + tag)
        if not isinstance(el, FieldMap):        # _def_fieldmap records the ladder's own entry
            self._record(rep, el, rule)
        return out

    def _directive(self, name: str, el: Directive, rule: Rule, rep: FidelityReport) -> list[str]:
        if el.format == "elegant" and el.card:
            self._expanded[el.name] = [name]
            rep.exact(el.name, "Directive", code="NATIVE_DIRECTIVE",
                      message=f"{el.card} re-emitted verbatim")
            joined = ", ".join(el.args)
            return [f"{name}: {el.card}" + (f", {joined}" if joined else "")]
        self._expanded[el.name] = []
        if el.role in self.COMMENT_ROLES:
            rep.exact(el.name, "Directive", code="DIRECTIVE_AS_COMMENT",
                      message=f"role {el.role!r} carries no elegant physics; written as a comment")
        else:
            rep.dropped(rule.code, rule.message, element=el.name, kind="Directive",
                        card=el.card, role=el.role)
        return [f"! lattix directive: {el.card} {' '.join(el.args)}".rstrip()]

    @staticmethod
    def _record(rep: FidelityReport, el: Element, rule: Rule) -> None:
        if rule.cls == "EXACT":
            rep.exact(el.name, el.kind)
        else:
            rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind,
                    target=rule.target)

    def _shift_attrs(self, el: Element, etype: str, rep: FidelityReport | None = None,
                     *, bend: bool = False) -> list[str]:
        """The misalignment attributes *etype* accepts; anything else is recorded LOSSY."""
        s = el.shift
        if s is None or s.is_zero():
            return []
        caps = _ALIGN_CAPS.get(etype.upper(), frozenset({"DX", "DY", "DZ"}))
        wanted = [("DX", s.x_offset), ("DY", s.y_offset), ("DZ", s.z_offset)]
        if bend:
            wanted += [("ETILT", s.tilt), ("EPITCH", s.x_rot), ("EYAW", s.y_rot)]
        else:
            wanted += [("TILT", s.tilt), ("PITCH", -s.x_rot), ("YAW", s.y_rot)]
        out, dropped = [], []
        for key, value in wanted:
            if not value:
                continue
            if key in caps:
                out.append(f"{key}={_num(value)}")
            else:
                dropped.append(key)
        if dropped and rep is not None:
            rep.lossy("MISALIGN_DROPPED",
                      f"elegant's {etype} has no {'/'.join(dropped)} attribute; that part of the "
                      "misalignment is not written",
                      element=el.name, kind=el.kind, elegant_type=etype, attrs=dropped)
        return out

    # per-kind builders: [(elegant type, [attribute strings]), …]
    def _def_drift(self, el, brho, rep):
        etype = self._pick(el, "Drift", "DRIF")
        return [(etype, [self._attr(el, "L", el.length, rep)]
                 + self._shift_attrs(el, etype, rep))]

    def _pick(self, el: Element, kind: str, default: str) -> str:
        src = self._source_type(el)
        return src if src in _KEEP_TYPE.get(kind, frozenset()) else default

    def _def_quadrupole(self, el, brho, rep):
        etype = self._pick(el, "Quadrupole", "KQUAD")
        attrs = [self._attr(el, "L", el.length, rep),
                 self._attr(el, "K1", el.multipole.Bn.get(1, 0.0) / brho, rep)]
        if el.multipole.Bn.get(2):
            attrs.append(self._attr(el, "K2", el.multipole.Bn[2] / brho, rep))
        if el.multipole.tilt.get(1):
            attrs.append(self._attr(el, "TILT", el.multipole.tilt[1], rep))
        attrs += self._shift_attrs(el, etype, rep)
        return [(etype, attrs)]

    def _def_sextupole(self, el, brho, rep):
        etype = self._pick(el, "Sextupole", "KSEXT")
        attrs = [self._attr(el, "L", el.length, rep),
                 self._attr(el, "K2", el.multipole.Bn.get(2, 0.0) / brho, rep)]
        if el.multipole.Bn.get(1):
            attrs.append(self._attr(el, "K1", el.multipole.Bn[1] / brho, rep))
        if el.multipole.Bs.get(1):
            attrs.append(self._attr(el, "J1", el.multipole.Bs[1] / brho, rep))
        if el.multipole.tilt.get(2):
            attrs.append(self._attr(el, "TILT", el.multipole.tilt[2], rep))
        attrs += self._shift_attrs(el, etype, rep)
        return [(etype, attrs)]

    def _def_octupole(self, el, brho, rep):
        etype = self._pick(el, "Octupole", "KOCT")
        attrs = [self._attr(el, "L", el.length, rep),
                 self._attr(el, "K3", el.multipole.Bn.get(3, 0.0) / brho, rep)]
        if el.multipole.tilt.get(3):
            attrs.append(self._attr(el, "TILT", el.multipole.tilt[3], rep))
        attrs += self._shift_attrs(el, etype, rep)
        return [(etype, attrs)]

    def _def_multipole(self, el, brho, rep):
        etype = self._pick(el, "Multipole", "MULT")
        table = dict(el.multipole.BnL)
        for order, value in el.multipole.BsL.items():
            if value:
                rep.lossy("SKEW_MULTIPOLE_DROPPED",
                          f"elegant MULT has no skew term; BsL[{order}] is not written",
                          element=el.name, kind="Multipole", order=order, BsL=value)
        if not table:
            table = {1: 0.0}
        out = []
        for order in sorted(table):
            attrs = [self._attr(el, "L", el.length, rep),
                     f"ORDER={order}",
                     self._attr(el, "KNL", table[order] / brho, rep)]
            if el.multipole.tilt.get(order):
                attrs.append(self._attr(el, "TILT", el.multipole.tilt[order], rep))
            attrs += self._shift_attrs(el, etype, rep)
            out.append((etype, attrs))
        if len(out) > 1:
            rep.equivalent("MULT_SPLIT_BY_ORDER",
                           f"an elegant MULT carries one order only; {len(out)} elements were "
                           "written, one per order, at the same position",
                           element=el.name, kind="Multipole", orders=sorted(table))
        return out

    def _def_bend(self, el, brho, rep):
        b = el.bend
        rect = b.rect
        etype = "RBEN" if rect else (self._source_type(el) if self._source_type(el) in _KEEP_BEND
                                     else "CSBEND")
        half = b.angle / 2.0 if rect else 0.0
        length = el.length
        if rect and b.angle:
            length = el.length * math.sin(half) / half        # arc -> chord
        attrs = [f"L={_num(length)}" if rect else self._attr(el, "L", el.length, rep),
                 self._attr(el, "ANGLE", b.angle, rep)]
        if (b.e1 - half) or (b.e2 - half):
            attrs.append(self._attr(el, "E1", b.e1 - half, rep))
            attrs.append(self._attr(el, "E2", b.e2 - half, rep))
        if b.edge_int2 is not None and etype in _SPLIT_FINT:
            attrs.append(f"FINT1={_num(b.edge_int1)}")
            attrs.append(f"FINT2={_num(b.edge_int2)}")
        else:
            if b.edge_int2 is not None and b.edge_int2 != b.edge_int1:
                rep.lossy("BEND_FINTX_DROPPED",
                          f"elegant {etype} has one FINT only; the exit fringe integral "
                          f"{b.edge_int2} is dropped",
                          element=el.name, kind="Bend", edge_int1=b.edge_int1,
                          edge_int2=b.edge_int2)
            if b.edge_int1 != DEFAULT_FINT:
                attrs.append(f"FINT={_num(b.edge_int1)}")
        if b.hgap:
            attrs.append(self._attr(el, "HGAP", b.hgap, rep))
        if el.multipole.Bn.get(1):
            attrs.append(self._attr(el, "K1", el.multipole.Bn[1] / brho, rep))
        if el.multipole.Bn.get(2):
            attrs.append(self._attr(el, "K2", el.multipole.Bn[2] / brho, rep))
        if b.tilt_ref:
            attrs.append(self._attr(el, "TILT", b.tilt_ref, rep))
        attrs += self._shift_attrs(el, etype, rep, bend=True)
        return [(etype, attrs)]

    def _def_solenoid(self, el, brho, rep):
        etype = self._pick(el, "Solenoid", "SOLE")
        attrs = [self._attr(el, "L", el.length, rep),
                 self._attr(el, "KS", el.solenoid.Bsol_T / brho, rep)]
        attrs += self._shift_attrs(el, etype, rep)
        return [(etype, attrs)]

    def _def_rfcavity(self, el, brho, rep):
        etype = self._pick(el, "RFCavity", "RFCA")
        rf = el.rf
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        sp = self._lattice.reference.species
        attrs = [self._attr(el, "L", el.length, rep),
                 self._attr(el, "VOLT", volt, rep),
                 self._phase_attr(el, rf.phase_rad),
                 self._attr(el, "FREQ", rf.frequency_Hz or 0.0, rep),
                 "CHANGE_P0=1"]
        rep.equivalent("ELEGANT_PHASE_FOR_SPECIES",
                       f"elegant's RF crest follows the charge sign: PHASE is written for "
                       f"{sp.name!r} (crest {'+90' if sp.charge < 0 else '-90'} deg); reading this "
                       "deck back as another species changes the physics",
                       element=el.name, kind="RFCavity", species=sp.name, charge=sp.charge,
                       phase_rad=rf.phase_rad)
        if abs(volt * math.cos(rf.phase_rad)) > _ACCEL_TOL_eV:
            rep.equivalent("RFCA_CHANGE_P0",
                           "CHANGE_P0=1 written so elegant's reference momentum follows the gain "
                           "(elegant's own default is 0)",
                           element=el.name, kind="RFCavity",
                           dE_eV=volt * math.cos(rf.phase_rad))
        if rf.n_cell:
            rep.lossy("NCELL_DROPPED",
                      "elegant RFCA has no cell count; n_cell is not written",
                      element=el.name, kind="RFCavity", n_cell=rf.n_cell)
        attrs += self._shift_attrs(el, etype, rep)
        return [(etype, attrs)]

    def _def_fieldmap(self, el, brho, rep):
        """PLAN §4.3 degradation ladder.  elegant's ``RFCA`` is a thin kick at the centre of a
        drift of the same length, so an RF map keeps its length in one element; a static
        solenoid/quadrupole map becomes a hard edge padded by two drifts (the writer's
        one-IR-element-to-many mechanism keeps them adjacent in the line)."""
        r = replacement_for(el)
        rep.add(r.cls, r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, message in r.extra:
            rep.add(cls, code, message, element=el.name, kind="FieldMap")
        out: list = []
        for part in r.parts:
            out.extend(getattr(self, f"_def_{part.kind.lower()}")(part, brho, rep))
        return out

    def _phase_attr(self, el: Element, phase_rad: float) -> str:
        """``PHASE=<source text>`` when the deck's own number is the same phase mod 360."""
        want = elegant_phase_deg(phase_rad, self._charge)
        if self._use_expressions:
            raw = self._native(el).get("attrs", {}).get("PHASE")
            got = evaluate_value(raw, self._values) if raw is not None else None
            if got is not None and abs((got - want + 180.0) % 360.0 - 180.0) <= 1e-9:
                return f"PHASE={raw}"
        return f"PHASE={_num(want)}"

    def _def_ncells(self, el, brho, rep):
        return [("DRIF", [f"L={_num(el.length)}"])]

    _def_rfqcell = _def_ncells

    def _def_kicker(self, el, brho, rep):
        if el.electric and el.hkick and el.vkick:
            rep.lossy("EKICK_AS_MAGNETIC",
                      "elegant's electric kickers are one plane each; a two-plane electric "
                      "steerer is written as a magnetic KICKER",
                      element=el.name, kind="Kicker")
        if el.electric and not el.vkick:
            return [("EHKICK", [self._attr(el, "L", el.length, rep),
                                self._attr(el, "KICK", el.hkick, rep)]
                     + self._shift_attrs(el, "EHKICK", rep))]
        if el.electric and not el.hkick:
            return [("EVKICK", [self._attr(el, "L", el.length, rep),
                                self._attr(el, "KICK", el.vkick, rep)]
                     + self._shift_attrs(el, "EVKICK", rep))]
        src = self._source_type(el)
        if src in ("HKICK", "HKIC") and not el.vkick:
            return [(src, [self._attr(el, "L", el.length, rep),
                           self._attr(el, "KICK", el.hkick, rep)]
                     + self._shift_attrs(el, src, rep))]
        if src in ("VKICK", "VKIC") and not el.hkick:
            return [(src, [self._attr(el, "L", el.length, rep),
                           self._attr(el, "KICK", el.vkick, rep)]
                     + self._shift_attrs(el, src, rep))]
        etype = src if src in ("KICKER", "KICK", "EKICKER") else "KICKER"
        return [(etype, [self._attr(el, "L", el.length, rep),
                         self._attr(el, "HKICK", el.hkick, rep),
                         self._attr(el, "VKICK", el.vkick, rep)]
                 + self._shift_attrs(el, etype, rep))]

    def _def_collimator(self, el, brho, rep):
        ap: ApertureP | None = el.aperture
        src = self._source_type(el)
        if src == "MAXAMP":
            attrs = [f"X_MAX={_num(ap.half_x or 0.0)}", f"Y_MAX={_num(ap.half_y or 0.0)}",
                     f"ELLIPTICAL={1 if ap.shape == 'ELLIPTICAL' else 0}"] if ap else []
            return [("MAXAMP", attrs + self._shift_attrs(el, "MAXAMP", rep))]
        etype = "ECOL" if (ap is not None and ap.shape == "ELLIPTICAL") else "RCOL"
        attrs = [self._attr(el, "L", el.length, rep)]
        if ap is not None:
            hx = ap.half_x if ap.half_x is not None else ap.half_y
            hy = ap.half_y if ap.half_y is not None else ap.half_x
            attrs.append(self._attr(el, "X_MAX", hx or 0.0, rep))
            attrs.append(self._attr(el, "Y_MAX", hy or 0.0, rep))
        attrs += self._shift_attrs(el, etype, rep)
        return [(etype, attrs)]

    def _def_marker(self, el, brho, rep):
        etype = self._pick(el, "Marker", "MARK")
        return [(etype, self._shift_attrs(el, etype, rep))]

    def _def_instrument(self, el, brho, rep):
        etype = _FAMILY_TYPE.get(el.family.upper())
        if etype is None:
            # elegant has no generic "instrument": a zero-length diagnostic is optically a
            # marker (nothing is lost but the family label, which stays in provenance), and
            # one with a length becomes a MONI so its drift space survives.
            if el.length:
                rep.equivalent("INSTRUMENT_AS_MONITOR",
                               f"elegant has no diagnostic type for family {el.family!r}; written "
                               "as MONI so the element's length is kept",
                               element=el.name, kind="Instrument", family=el.family,
                               length=el.length)
                return [("MONI", [self._attr(el, "L", el.length, rep)]
                         + self._shift_attrs(el, "MONI", rep))]
            rep.equivalent("INSTRUMENT_AS_MARKER",
                           f"elegant has no diagnostic type for family {el.family!r}; the "
                           "zero-length diagnostic is written as MARK (same optics, the family "
                           "label survives only in provenance)",
                           element=el.name, kind="Instrument", family=el.family)
            return [("MARK", self._shift_attrs(el, "MARK", rep))]
        if etype == "WATCH":
            fn = el.params.get("filename") or f"%s.{el.name.lower()}"
            return [("WATCH", [f'FILENAME="{fn}"'] + self._shift_attrs(el, "WATCH", rep))]
        return [(etype, [self._attr(el, "L", el.length, rep)]
                 + self._shift_attrs(el, etype, rep))]

    def _def_foil(self, el, brho, rep):
        return [("MARK", [])]

    def _def_taylor(self, el, brho, rep):
        attrs = [self._attr(el, "L", el.length, rep), "ORDER=1"]
        for i in range(6):
            for j in range(6):
                v = el.matrix[i][j]
                if v != (1.0 if i == j else 0.0):
                    attrs.append(f"R{i + 1}{j + 1}={_num(v)}")
        attrs += [f"C{i + 1}={_num(v)}" for i, v in enumerate(el.offset) if v]
        attrs += self._shift_attrs(el, "EMATRIX", rep)
        return [("EMATRIX", attrs)]

    def _def_patch(self, el, brho, rep):
        return [("MARK", [])]

    def _def_referencechange(self, el, brho, rep):
        return [("MARK", [])]

    def _def_superposition(self, el, brho, rep):     # pragma: no cover - expanded in _items
        return [("MARK", [])]

    # -- beam lines ---------------------------------------------------------
    def _line_block(self, root_name: str, line_names: dict[str, str], lattice: Lattice,
                    defs: dict, rep: FidelityReport) -> list[str]:
        by_name = {el.name: nm for nm, el, _ in defs.values()}
        root = lattice.use or lattice.name
        if root not in lattice.lines:
            rep.equivalent("NO_LINE_STRUCTURE",
                           f"the lattice has no line named {root!r}; a flat beam line was written",
                           element=None, kind=None)
            flat = ", ".join(nm for p in lattice.flatten()
                             for nm in self._expanded.get(p.element.name, []))
            return _wrap(f"{root_name}: LINE=({flat})")

        sup_lines: list[str] = []
        split_lines_seen: set[str] = set()
        nil: list[str] = []

        def emits_nothing(ref: str) -> bool:
            el = lattice.elements.get(ref)
            return isinstance(el, Freq) or (isinstance(el, Directive)
                                            and not (el.format == "elegant" and el.card))

        def nil_marker() -> str:
            if not nil:
                nil.append(f"{root_name}_NIL")
            return nil[0]

        def ref_name(ref: str) -> str:
            if ref in line_names:
                return line_names[ref]
            parts = self._expanded.get(ref)
            if parts and len(parts) > 1:      # a MULT split by order: keep all the pieces
                nm = f"{parts[0]}__ALL"
                if nm not in split_lines_seen:
                    split_lines_seen.add(nm)
                    sup_lines.extend(_wrap(f"{nm}: LINE=({', '.join(parts)})"))
                return nm
            if ref in by_name:
                return by_name[ref]
            el = lattice.elements.get(ref)
            if isinstance(el, Superposition):      # children were emitted, the parent was not
                nm = f"{ref.upper()}_SUP".replace("$", "_")
                body = ", ".join(by_name[c] for _, c in el.children if c in by_name)
                sup_lines.extend(_wrap(f"{nm}: LINE=({body})"))
                return nm
            return ref.upper()

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
            emitted.extend(_wrap(f"{line_names.get(name, name.upper())}: LINE=({body})"))

        emit(root)
        head = [f"{nil[0]}: MARK"] if nil else []
        return head + sup_lines + emitted


def _wrap(text: str, width: int = 100) -> list[str]:
    """Split a long ``name: LINE=(…)`` over continuation lines with a trailing ``&``."""
    if len(text) <= width:
        return [text]
    out: list[str] = []
    cur = ""
    for piece in text.split(", "):
        candidate = piece if not cur else f"{cur}, {piece}"
        if len(candidate) > width and cur:
            out.append(cur + ",&")
            cur = piece
        else:
            cur = candidate
    out.append(cur)
    return out


_MISSING = set(ALL_KINDS) - set(Writer.RULES)
if _MISSING:                                        # pragma: no cover - guarded by a test too
    raise RuntimeError(f"elegant writer RULES do not cover {sorted(_MISSING)}")
