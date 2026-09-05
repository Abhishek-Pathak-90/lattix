"""PALS writer (PLAN §6 task 2.4): table-driven, never silent.

Emits a document whose shape follows ``source/fundamentals.md`` of
<https://github.com/pals-project/pals> at commit ``a2b1083`` (2026-09-01)::

    PALS:
      version: null
      notes: ["… lattix <version> …"]
      facility:
        - <element definitions>          # one YAML key per element, `kind:` + parameter groups
        - <BeamLine definitions>         # `line:` items, `repeat:`/`direction:`
        - <Lattice>                      # `branches: [root]`
        - use: <lattice>

Conventions this writer fixes (all from the standard, see :mod:`lattix.formats.pals.reader`
for the full IR ⇄ PALS name table):

* **fields, not normalized strengths** — ``MagneticMultipoleP.Bn1`` [T/m], ``SolenoidP.Bsol``
  [T] — because the IR's canonical strengths are lab fields and a field needs no reference
  momentum; ``normalized=True`` switches to ``Kn1``/``Ksol`` computed with the *signed*
  rigidity at each element's entrance (and then a source expression such as MAD-X's
  ``k1 := kqf`` can be re-emitted verbatim);
* ``RFP.phase`` in rad/2π via :func:`lattix.ir.rf.pals_phase` with
  ``zero_phase: ACCELERATING`` (the IR's 0 = crest), and ``dE_ref`` written **explicitly**
  from :func:`lattix.ir.walk.energy_gain_eV` — PALS wants the reference energy change stated;
* a bend is given by ``length`` + ``BendP.angle_ref`` (one parameter from the length set and
  one from the angle set, which is exactly what ``parameters/bend.md`` allows) with
  ``e1``/``e2`` for a sector source and ``e1_rect``/``e2_rect`` for a rectangular one, and
  ``edge1_int = fint·hgap`` [m] — PALS has no separate ``hgap``;
* a ``Kicker``'s deflections become order-0 integrated multipoles ``Bn0L = −hkick·Bρ``,
  ``Bs0L = +vkick·Bρ`` (sign derivation in
  :func:`lattix.formats.pals.reader._kicks_from`);
* the root ``BeamLine`` starts with a ``BeginningEle`` carrying
  ``ReferenceP {species_ref, E_tot_ref}`` — a branch must start with one
  (``lattice-construction.md`` §Branch Expansion).

Three ``flavor``s, all valid PALS:

``"standard"`` (default)
    as above: element definitions at ``facility`` level, the IR's nested ``Line``s as
    ``BeamLine``s with ``repeat:``/``direction:``, a ``Lattice`` and a ``use:``;
``"flat"``
    the same ``PALS:`` root but **one** root ``BeamLine`` holding the fully expanded lattice
    with every occurrence defined in place under its own name — what a consumer that cannot
    expand sublines or ``repeat:`` counts needs;
``"beamline"``
    the *pre-standard* bare one-key ``BeamLine`` document with no ``PALS:`` root at all
    (``pals-schema`` 0.2 / ImpactX ≤ 25 read this through ``BeamLine.from_file``; ImpactX
    ships one as ``examples/pals/fodo.pals.yaml``).  ``pals-schema`` 0.3's ``PALSroot`` no
    longer loads it.

``beginning=False`` leaves the ``BeginningEle`` out (recorded LOSSY: the reference particle
then has nowhere to live).  ImpactX 26.08's ``pals_to_impactx.read_lattice`` raises on every
kind but ``Drift``/``Quadrupole`` — ``BeginningEle`` included — so its oracle test uses
``flavor="flat", beginning=False``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.pals.reader import ir_species, pals_species_name
from lattix.ir.elements import (
    ApertureP,
    Bend,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Multipole,
    NCells,
    Patch,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Solenoid,
    Superposition,
    Taylor,
)
from lattix.ir.expr import ExpressionError, evaluate
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Line, LineItem
from lattix.ir.reference import ReferenceParticle
from lattix.ir.rf import pals_phase
from lattix.ir.walk import energy_gain_eV, propagate

#: ``ApertureP.aperture_at`` → PALS ``ApertureP.location``.
_APERTURE_LOCATION = {"ENTRANCE": "ENTRANCE_END", "EXIT": "EXIT_END",
                      "BOTH_ENDS": "BOTH_ENDS", "CONTINUOUS": "EVERYWHERE"}

#: ``Directive.role``s PALS can carry without losing physics.
_MARKER_ROLES = frozenset({"period_start", "period_end"})
_NOTE_ROLES = frozenset({"title"})

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")     # fundamentals.md §Names
_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_EXPR_TOL = 1e-12
_TOL = 1e-15


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str                 # the PALS kind this IR kind becomes
    cls: str = "EXACT"
    code: str = "OK"
    message: str = ""


def sanitize(name: str) -> str:
    """A PALS name: letter or ``_`` first, then alphanumerics and ``_`` (case is kept)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())
    if not s or not (s[0].isalpha() or s[0] == "_"):
        s = "e_" + s
    return s


class NameMap:
    """Sanitize + uniquify; remembers renames so the writer can emit ``MetaP.alias``."""

    def __init__(self) -> None:
        self._by_key: dict[int, str] = {}
        self._used: set[str] = set()
        self.renamed: dict[str, str] = {}

    def assign(self, original: str, key: object | None = None) -> str:
        if key is not None and id(key) in self._by_key:
            return self._by_key[id(key)]
        base = sanitize(original)
        name, k = base, 2
        while name in self._used:
            name = f"{base}_{k}"
            k += 1
        self._used.add(name)
        if key is not None:
            self._by_key[id(key)] = name
        if name != original:
            self.renamed[name] = original
        return name

    def reserve(self, name: str) -> str:
        return self.assign(name)


def _f(x: float) -> float | str | None:
    """A YAML/JSON-safe number: ``-0.0`` normalised, infinities as PALS's own ``Inf``/``-Inf``
    symbols (``fundamentals.md`` §Special Values); the float repr keeps full precision."""
    v = float(x)
    if v != v:
        return None
    if v == float("inf"):
        return "Inf"
    if v == float("-inf"):
        return "-Inf"
    return 0.0 if v == 0.0 else v


@dataclass
class _Item:
    element: Element
    ref: ReferenceParticle
    dE: float


class Writer:
    """``Writer().write(lattice, path)`` → :class:`~lattix.fidelity.FidelityReport`."""

    format = "pals"

    RULES: dict[str, Rule] = {
        "Drift": Rule("Drift"),
        "Quadrupole": Rule("Quadrupole"),
        "Sextupole": Rule("Sextupole"),
        "Octupole": Rule("Octupole"),
        "Multipole": Rule("Multipole"),
        "Bend": Rule("Bend"),
        "Solenoid": Rule("Solenoid"),
        "RFCavity": Rule("RFCavity"),
        "FieldMap": Rule("RFCavity/UnionEle/Drift", "EQUIVALENT", "FM_TO_CAVITY",
                         "field map written as an RFCavity with an explicit dE_ref"),
        "NCells": Rule("Drift", "LOSSY", "NCELLS_TO_DRIFT",
                       "PALS has no multi-cell DTL/CCL element; written as a drift"),
        "RFQCell": Rule("Drift", "LOSSY", "RFQ_TO_DRIFT",
                        "PALS has no RFQ cell element; written as a drift"),
        "Kicker": Rule("Kicker"),
        "Collimator": Rule("Mask"),
        "Marker": Rule("Marker"),
        "Instrument": Rule("Instrument"),
        "Foil": Rule("Foil"),
        "Taylor": Rule("Taylor"),
        "Patch": Rule("Patch"),
        "ReferenceChange": Rule("ReferenceChange"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each cavity's RFP.frequency"),
        "Directive": Rule("MetaP note", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive kept only as a PALS note"),
        "Superposition": Rule("UnionEle", "EQUIVALENT", "SUPERPOSITION_AS_UNIONELE",
                              "overlapping fields written as a UnionEle of shifted children"),
    }

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False, format: str = "auto",
              expressions: bool = True, flavor: str = "standard", normalized: bool = False,
              beginning: bool = True, lattice_name: str | None = None) -> FidelityReport:
        """Write *lattice* as a PALS document.

        ``format`` is ``"yaml"``, ``"json"`` or ``"auto"`` (the path suffix, YAML unless
        ``.json``).  ``expressions`` re-emits the IR's source expressions where they still
        evaluate to the number being written.  ``normalized`` writes ``Kn{n}``/``Ksol``
        instead of the lab fields.  ``beginning=False`` omits the ``BeginningEle``.
        ``flavor`` is ``"standard"``, ``"flat"`` or ``"beamline"`` (see the module docstring).
        """
        if flavor not in ("standard", "flat", "beamline"):
            raise ValueError(f"flavor must be 'standard', 'flat' or 'beamline', got {flavor!r}")
        as_json = format == "json" or (format == "auto" and Path(path).suffix.lower() == ".json")
        if format not in ("auto", "yaml", "json"):
            raise ValueError(f"format must be 'auto', 'yaml' or 'json', got {format!r}")

        self.rep = FidelityReport(target_format="pals", target_file=str(path))
        self.lat = lattice
        self.names = NameMap()
        self.notes: list[str] = [f"written by lattix {__version__}"
                                 f" from {lattice.meta.get('source_format', 'the lattix IR')}"]
        self.normalized = normalized
        self.expressions = expressions
        self.beginning = beginning and flavor != "beamline"
        if not self.beginning:
            self.rep.lossy("PALS_NO_BEGINNING_ELE",
                           "no BeginningEle is written, so the reference species and energy are "
                           "not in the document (a PALS branch must start with one)",
                           species=lattice.reference.species.name,
                           E_tot_ref=lattice.reference.total_energy_eV)
        self.variables = {k: v.value for k, v in lattice.variables.items()}

        placed = propagate(lattice)
        self.lines: dict[str, Line] = dict(lattice.lines)
        self.root = lattice.use or lattice.name
        if self.root not in self.lines:                 # a lattice with no Line: expand it
            self.root = sanitize(self.root or "main")
            self.lines[self.root] = Line(name=self.root, items=[
                LineItem(ref=p.element.name, reverse=p.reversed) for p in placed])
        self.first: dict[int, _Item] = {}
        for p in placed:
            ref = p.ref_in or lattice.reference
            item = _Item(p.element, ref, energy_gain_eV(p.element, ref))
            prev = self.first.get(id(p.element))
            if prev is None:
                self.first[id(p.element)] = item
                continue
            rigid = normalized and abs(prev.ref.brho_signed - ref.brho_signed) > \
                1e-12 * abs(prev.ref.brho_signed)
            if rigid or abs(prev.dE - item.dE) > 1e-6:
                self.rep.equivalent(
                    "PALS_MULTI_ENERGY_DEFINITION",
                    "this definition is used at two reference energies; RFP.dE_ref (and any "
                    "normalized strength) uses the first occurrence",
                    element=p.element.name, kind=p.element.kind,
                    dE_first=prev.dE, dE_here=item.dE)

        doc = (self._beamline_document(placed) if flavor == "beamline"
               else self._standard_document(lattice_name, placed if flavor == "flat" else None))
        text = (json.dumps(doc, indent=2, allow_nan=False) + "\n" if as_json else
                _header(lattice, flavor) +
                yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100,
                               allow_unicode=True))
        Path(path).write_text(text, encoding="utf-8")
        self.rep.raise_if(strict)
        return self.rep

    # -- document shells ----------------------------------------------------
    def _standard_document(self, lattice_name: str | None, flat: list | None = None) -> dict:
        root = self.root
        facility: list[Any] = []
        if flat is not None:
            facility.extend(self._variable_entries())
            root = self.names.reserve(sanitize(root or "main"))
            facility.append({root: self._flat_beamline(flat, is_root=True)})
            return self._wrap(facility, root, lattice_name)
        lines = self._line_order(root)
        refs: list[str] = []
        for name in lines:
            refs.extend(it.ref for it in self.lines[name].items)

        facility.extend(self._variable_entries())
        skip: set[str] = set()
        union_children: set[str] = set()
        for el in self.lat.elements.values():
            if isinstance(el, Superposition):
                union_children.update(n for _o, n in el.children)
        seen: set[str] = set()
        for name in refs:
            if name in seen or name in union_children or name in self.lines:
                continue
            seen.add(name)
            el = self.lat.elements.get(name)
            if el is None:
                self.rep.dropped("PALS_UNDEFINED_ELEMENT", f"line references undefined {name!r}",
                                 element=name)
                skip.add(name)
                continue
            key = self.names.assign(el.name, el)
            node = self._definition(el, key)
            if node is None:
                skip.add(name)
                continue
            facility.append({key: node})

        line_names = {n: self.names.assign(n) for n in lines}
        for name in lines:
            facility.append({line_names[name]: self._beamline(name, line_names, skip,
                                                              is_root=(name == root))})
        return self._wrap(facility, line_names[root], lattice_name)

    def _wrap(self, facility: list[Any], root: str, lattice_name: str | None) -> dict:
        lat_name = self.names.reserve(
            lattice_name or self.lat.meta.get("pals_lattice")
            or f"{sanitize(self.lat.name or 'lattix')}_lattice")
        facility.append({lat_name: {"kind": "Lattice", "branches": [root]}})
        facility.append({"use": lat_name})

        doc: dict[str, Any] = {"version": None}
        authors = self.lat.meta.get("pals_authors")
        if authors:
            doc["authors"] = authors
        doc["notes"] = self.notes + [n for n in (self.lat.meta.get("pals_notes") or [])
                                     if not str(n).startswith("written by lattix")]
        for key in ("reminders", "extension_labels", "phase_space_coordinates"):
            if self.lat.meta.get(f"pals_{key}") is not None:
                doc[key] = self.lat.meta[f"pals_{key}"]
        if getattr(self, "_custom_species", False) and "lattix" not in (doc.get("extension_labels") or []):
            doc["extension_labels"] = [*(doc.get("extension_labels") or []), "lattix"]
        doc["facility"] = facility
        return {"PALS": doc}

    def _beamline_document(self, placed) -> dict:
        """The bare one-key ``BeamLine`` shape (pals-schema 0.2 / ImpactX ≤ 25); fully flattened."""
        self.rep.equivalent("PALS_BEAMLINE_FLAVOR",
                            "written in the pre-standard bare-BeamLine shape: no 'PALS:' root, "
                            "no Lattice and no use statement")
        name = self.names.reserve(sanitize(self.root or "main"))
        return {name: self._flat_beamline(placed, is_root=False)}

    def _flat_beamline(self, placed, *, is_root: bool) -> dict:
        """One BeamLine holding the expanded lattice, every occurrence defined in place."""
        self.rep.equivalent("PALS_FLATTENED",
                            "the lattice is written as one expanded BeamLine: sublines, repeat "
                            "counts and shared element definitions are resolved away",
                            occurrences=len(placed))
        items: list[Any] = []
        if is_root and self.beginning:
            items.append({self.names.reserve(self.lat.meta.get("pals_beginning") or "begin"):
                          self._beginning()})
        for p in placed:
            key = self.names.assign(p.element.name)      # unkeyed: unique per occurrence
            node = self._definition(p.element, key)
            if node is None:
                continue
            if p.reversed:                # beamlines.md: an in-place item may carry `direction`
                node["direction"] = -1
            items.append({key: node})
        return {"kind": "BeamLine", "line": items}

    # -- lines --------------------------------------------------------------
    def _line_order(self, root: str) -> list[str]:
        """Lines reachable from *root*, sublines before the lines that use them."""
        if root not in self.lines:
            return []
        out: list[str] = []
        seen: set[str] = set()

        def visit(name: str, stack: frozenset[str]) -> None:
            if name in seen or name in stack or name not in self.lines:
                return
            for it in self.lines[name].items:
                visit(it.ref, stack | {name})
            seen.add(name)
            out.append(name)

        visit(root, frozenset())
        return out

    def _beamline(self, name: str, line_names: dict[str, str], skip: set[str],
                  *, is_root: bool) -> dict:
        items: list[Any] = []
        if is_root and self.beginning:
            items.append({self.names.reserve(self.lat.meta.get("pals_beginning") or "begin"):
                          self._beginning()})
        for it in self.lines[name].items:
            if it.ref in skip:
                continue
            ref = line_names.get(it.ref) or self.names.assign(
                it.ref, self.lat.elements.get(it.ref))
            body: dict[str, Any] = {}
            if it.repeat != 1:
                body["repeat"] = int(it.repeat)
            if it.reverse:
                body["direction"] = -1
            items.append({ref: body} if body else ref)
        node: dict[str, Any] = {"kind": "BeamLine", "line": items}
        if name in (self.lat.meta.get("pals_periodic") or []):
            node["periodic"] = True
        return node

    def _beginning(self) -> dict:
        ref = self.lat.reference
        node: dict[str, Any] = {
            "kind": "BeginningEle",
            "ReferenceP": {"species_ref": pals_species_name(ref.species),
                           "E_tot_ref": _f(ref.total_energy_eV)},
        }
        if ref.time_s:
            node["ReferenceP"]["time_ref"] = _f(ref.time_s)
        node.update(self.lat.meta.get("pals_beginning_groups") or {})
        if ir_species(node["ReferenceP"]["species_ref"]) is None:
            # not a species PALS names: keep mass and charge in a namespaced extension block so
            # the reference particle (and every normalized strength) survives a round trip
            sp = ref.species
            node["lattix"] = {"species": {"name": sp.name, "mass_eV": _f(sp.mass_eV), "charge": sp.charge}}
            self._custom_species = True
            self.rep.equivalent("PALS_CUSTOM_SPECIES_EXTENSION",
                                f"species {sp.name!r} is not a PALS species name; mass and charge are "
                                "written in the 'lattix' extension block of the BeginningEle",
                                element=None, kind=None)
        return node

    # -- variables ----------------------------------------------------------
    def _variable_entries(self) -> list[dict]:
        if not self.expressions or not self.lat.variables:
            return []
        keep = {}
        for name, var in self.lat.variables.items():
            if _NAME_RE.match(name):
                keep[name] = var
            else:
                self.rep.equivalent("VARIABLE_DROPPED",
                                    f"variable {name!r} is not a writable PALS name; its value is "
                                    "folded into the numbers", variable=name)
        if not keep:
            return []
        for name in keep:
            self.names.reserve(name)
        items = []
        for name in _topo(keep):
            var = keep[name]
            expr = var.expression
            text = expr.text.strip() if (expr is not None and expr.text.strip()) else None
            items.append({name: text if text and self._checks(text, var.value) else _f(var.value)})
        return [{"variables": items}]

    def _checks(self, text: str, value: float) -> bool:
        try:
            return abs(evaluate(text, self.variables) - value) <= _EXPR_TOL * max(1.0, abs(value))
        except ExpressionError:
            return False

    def _maybe_expr(self, el: Element, ir_path: str, value: float) -> Any:
        """The IR's source expression when it still evaluates to *value*, else the number."""
        if self.expressions:
            expr = el.expressions.get(ir_path)
            if expr is not None and expr.text.strip() and self._checks(expr.text, value):
                return expr.text.strip()
        return _f(value)

    # -- element definitions -------------------------------------------------
    def _definition(self, el: Element, name: str, parent: _Item | None = None) -> dict | None:
        rule = self.RULES.get(el.kind)
        if rule is None:                                    # pragma: no cover - RULES is total
            raise KeyError(f"PALS writer has no rule for kind {el.kind!r}")
        if isinstance(el, Freq):
            self.rep.exact(el.name, "Freq", message=rule.message)
            return None
        if isinstance(el, Directive):
            return self._directive(el, rule)
        item = self.first.get(id(el))
        if item is None:                     # e.g. a UnionEle child: never placed on its own
            ref = parent.ref if parent is not None else self.lat.reference
            item = _Item(el, ref, energy_gain_eV(el, ref))
        node = getattr(self, f"_def_{el.kind.lower()}")(el, item)
        if node is None:
            return None
        kind = node.pop("kind")
        out: dict[str, Any] = {"kind": kind}
        if el.length or kind not in ("Marker", "Patch", "ReferenceChange", "Taylor",
                                     "BeginningEle"):
            out["length"] = self._maybe_expr(el, "length", el.length)
        out.update(node)
        self._common_groups(el, name, out)
        if not isinstance(el, FieldMap):        # _def_fieldmap records the ladder's own entry
            self._record(el, rule)
        return out

    def _record(self, el: Element, rule: Rule) -> None:
        if rule.cls == "EXACT":
            self.rep.exact(el.name, el.kind)
        else:
            self.rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind,
                         target=rule.target)

    def _common_groups(self, el: Element, name: str, out: dict) -> None:
        meta = dict(el.meta.get("MetaP") or {})
        original = (el.provenance.original_name if el.provenance else None) or el.name
        if name != original:
            meta["alias"] = original
        if isinstance(el, Instrument) and el.family and el.family != "MONITOR":
            meta["label"] = el.family
        if meta:
            out["MetaP"] = meta
        if el.aperture is not None and not isinstance(el, Collimator):
            ap = self._aperture(el)
            if ap:
                out["ApertureP"] = ap
        if el.shift is not None and not el.shift.is_zero():
            out["BodyShiftP"] = _body_shift(el.shift)
        if el.tracking:
            out["TrackingP"] = dict(el.tracking)
        for key in ("FloorP",):
            if el.meta.get(key):
                out[key] = el.meta[key]
        for key, value in (el.native.get("pals", {}).get("groups") or {}).items():
            out.setdefault(key, value)

    def _aperture(self, el: Element) -> dict:
        ap: ApertureP = el.aperture
        node: dict[str, Any] = {}
        for axis, limits in (("x", ap.x_limits), ("y", ap.y_limits)):
            if limits is not None:
                node[f"{axis}_min"], node[f"{axis}_max"] = _f(limits[0]), _f(limits[1])
        if not node:
            return {}
        node["shape"] = ap.shape
        node["location"] = _APERTURE_LOCATION.get(ap.aperture_at, "BOTH_ENDS")
        return node

    def _multipole_group(self, el: Element, item: _Item, mp: MagneticMultipoleP,
                         integrated: bool = False) -> dict:
        """``Bn{n}``/``Bs{n}`` (or ``Kn{n}``/``Ks{n}`` with ``normalized=True``), plus
        ``tilt{n}``; an ``L`` suffix marks the length-integrated form.

        A PALS source that wrote a normalized component gets it back in the same form:
        ``native["pals"]["multipole_form"]`` remembers it per order and side, which is what
        ``native[fmt]`` is for (PLAN §4.1) and keeps a pals→pals diff small."""
        brho = item.ref.brho_signed
        source_form = el.native.get("pals", {}).get("multipole_form", {})
        out: dict[str, Any] = {}
        tables = ((mp.Bn, "n", False), (mp.Bs, "s", False), (mp.BnL, "n", True),
                  (mp.BsL, "s", True))
        for table, side, is_int in tables:
            for order in sorted(table):
                value = table[order]
                if not value:
                    continue
                suffix = "L" if is_int or integrated else ""
                if self.normalized or source_form.get(f"{side}{order}") == "K":
                    key = f"K{side}{order}{suffix}"
                    number = value / brho
                else:
                    key = f"B{side}{order}{suffix}"
                    number = value
                path = f"multipole.{'BnL' if is_int else 'Bn'}[{order}]" if side == "n" else \
                       f"multipole.{'BsL' if is_int else 'Bs'}[{order}]"
                out[key] = self._maybe_expr(el, path, number) if key[0] == "K" else _f(number)
        for order in sorted(mp.tilt):
            if mp.tilt[order]:
                out[f"tilt{order}"] = _f(mp.tilt[order])
        return out

    # -- per-kind builders ---------------------------------------------------
    def _def_drift(self, el: Drift, item: _Item) -> dict:
        return {"kind": "Drift"}

    def _def_quadrupole(self, el: Quadrupole, item: _Item) -> dict:
        return {"kind": "Quadrupole",
                "MagneticMultipoleP": self._multipole_group(el, item, el.multipole)}

    def _def_sextupole(self, el, item: _Item) -> dict:
        return {"kind": "Sextupole",
                "MagneticMultipoleP": self._multipole_group(el, item, el.multipole)}

    def _def_octupole(self, el, item: _Item) -> dict:
        return {"kind": "Octupole",
                "MagneticMultipoleP": self._multipole_group(el, item, el.multipole)}

    def _def_multipole(self, el: Multipole, item: _Item) -> dict:
        return {"kind": "Multipole",
                "MagneticMultipoleP": self._multipole_group(el, item, el.multipole)}

    def _def_bend(self, el: Bend, item: _Item) -> dict:
        b = el.bend
        # `length` is one of the length-set parameters, so exactly one more may be given:
        # the angle, or a curvature if that is how the source spelled it.
        form = el.native.get("pals", {}).get("bend_form") or []
        if "angle_ref" not in form and {"g_ref", "radius_ref", "rho_ref"} & set(form):
            bend: dict[str, Any] = {"g_ref": _f(b.g_ref(el.length))}
        else:
            bend = {"angle_ref": self._maybe_expr(el, "bend.angle", b.angle)}
        if b.rect:
            bend["e1_rect"] = _f(b.e1 - b.angle / 2)
            bend["e2_rect"] = _f(b.e2 - b.angle / 2)
        else:
            bend["e1"], bend["e2"] = _f(b.e1), _f(b.e2)
        if b.edge_int1 or b.edge_int2:
            if b.edge_int1 * b.hgap:
                bend["edge1_int"] = _f(b.edge_int1 * b.hgap)
            fintx = b.edge_int1 if b.edge_int2 is None else b.edge_int2
            if fintx * b.hgap:
                bend["edge2_int"] = _f(fintx * b.hgap)
            if b.hgap == 0.0 and (b.edge_int1 or b.edge_int2):
                self.rep.lossy("PALS_EDGE_INT_NEEDS_HGAP",
                               "PALS stores only the product fint·hgap and this bend has hgap = 0, "
                               "so the fringe-field integral is written as zero",
                               element=el.name, kind="Bend", fint=b.edge_int1, hgap=b.hgap)
        if b.tilt_ref:
            bend["tilt_ref"] = _f(b.tilt_ref)
        if b.fringe_k2 is not None:
            self.rep.lossy("PALS_FRINGE_K2_DROPPED",
                           "the second-order fringe coefficient (TraceWin EDGE K2) has no PALS "
                           "field", element=el.name, kind="Bend", fringe_k2=b.fringe_k2)
        out: dict[str, Any] = {"kind": "Bend", "BendP": bend}
        mp = self._multipole_group(el, item, el.multipole)
        if mp:
            out["MagneticMultipoleP"] = mp
        return out

    def _def_solenoid(self, el: Solenoid, item: _Item) -> dict:
        if self.normalized or el.native.get("pals", {}).get("solenoid_form") == "Ksol":
            group = {"Ksol": _f(el.solenoid.Bsol_T / item.ref.brho_signed)}
        else:
            group = {"Bsol": _f(el.solenoid.Bsol_T)}
        out: dict[str, Any] = {"kind": "Solenoid", "SolenoidP": group}
        mp = self._multipole_group(el, item, el.multipole) if hasattr(el, "multipole") else {}
        if mp:
            out["MagneticMultipoleP"] = mp
        return out

    def _def_rfcavity(self, el: RFCavity, item: _Item) -> dict:
        return {"kind": "RFCavity", "RFP": self._rf_group(el, el.rf, item)}

    def _def_fieldmap(self, el: FieldMap, item: _Item) -> dict:
        """PLAN §4.3 degradation ladder.  The map files are not part of the PALS standard, so an
        RF map becomes an ``RFCavity`` with an explicit ``dE_ref`` and a static solenoid /
        quadrupole map a hard edge; a hard edge shorter than the map is wrapped in a
        ``UnionEle`` of the map's own length so the padding stays implicit and the survey
        does not move."""
        r = replacement_for(el)
        self.rep.add(r.cls, r.code, r.message, element=el.name, kind="FieldMap",
                     **{"files": list(el.files), **r.details})
        for cls, code, message in r.extra:
            self.rep.add(cls, code, message, element=el.name, kind="FieldMap")
        body = r.main
        if body.kind == "Drift":
            return {"kind": "Drift"}
        if body.kind == "RFCavity":
            return {"kind": "RFCavity", "RFP": self._rf_group(body, body.rf, item)}
        if not r.padded:
            return getattr(self, f"_def_{body.kind.lower()}")(body, item)
        key = self.names.assign(f"{el.name}_core", body)
        node = self._definition(body, key, parent=item)
        return {"kind": "UnionEle", "elements": {key: node}}

    def _def_ncells(self, el: NCells, item: _Item) -> dict:
        self.rep.lossy("NCELLS_TO_DRIFT",
                       "PALS has no multi-cell DTL/CCL element; written as a drift of the same "
                       "length", element=el.name, kind="NCells", dE_ref=item.dE)
        return {"kind": "Drift"}

    def _def_rfqcell(self, el: RFQCell, item: _Item) -> dict:
        self.rep.lossy("RFQ_TO_DRIFT", "PALS has no RFQ cell element; written as a drift of the "
                       "same length", element=el.name, kind="RFQCell")
        return {"kind": "Drift"}

    def _rf_group(self, el: Element, rf, item: _Item) -> dict:
        out: dict[str, Any] = {}
        if rf.frequency_Hz:
            out["frequency"] = _f(rf.frequency_Hz)
        active = rf.L_active_m if rf.L_active_m is not None else el.length
        use_gradient = (el.native.get("pals", {}).get("rf_form") == "gradient"
                        and rf.gradient_V_per_m is not None)
        if use_gradient:
            out["gradient"] = _f(rf.gradient_V_per_m)
        else:
            out["voltage"] = _f(rf.voltage_V)
        out["phase"] = _f(pals_phase(rf.phase_rad))
        out["zero_phase"] = "ACCELERATING"
        if rf.cavity_type != "STANDING_WAVE":
            out["cavity_type"] = rf.cavity_type
        if rf.L_active_m is not None and abs(active - el.length) > _TOL:
            out["L_active"] = _f(active)
        if rf.n_cell:
            out["num_cells"] = int(rf.n_cell)
        out["dE_ref"] = _f(rf.dE_ref_eV if rf.dE_ref_eV is not None else item.dE)
        if rf.ttf != 1.0:
            self.rep.lossy("PALS_TTF_DROPPED",
                           f"transit-time factor {rf.ttf:g} has no PALS field (it is already "
                           "folded into the voltage)", element=el.name, kind=el.kind, ttf=rf.ttf)
        if not rf.phase_is_sync:
            self.rep.equivalent("PALS_PHASE_AS_SYNC",
                                "the IR phase is a deck RF phase, not a synchronous phase; it is "
                                "written as RFP.phase with zero_phase = ACCELERATING",
                                element=el.name, kind=el.kind)
        return out

    def _def_kicker(self, el: Kicker, item: _Item) -> dict:
        brho = item.ref.brho_signed
        group: dict[str, Any] = {}
        if self.normalized:
            if el.hkick:
                group["Kn0L"] = _f(-el.hkick)
            if el.vkick:
                group["Ks0L"] = _f(el.vkick)
        else:
            if el.hkick:
                group["Bn0L"] = _f(-el.hkick * brho)
            if el.vkick:
                group["Bs0L"] = _f(el.vkick * brho)
        if el.electric:
            self.rep.lossy("ELECTRIC_KICKER_AS_MAGNETIC",
                           "an electrostatic steerer is written as a magnetic order-0 multipole "
                           "with the same deflection", element=el.name, kind="Kicker")
        out: dict[str, Any] = {"kind": "Kicker"}
        if group:
            out["MagneticMultipoleP"] = group
        return out

    def _def_collimator(self, el: Collimator, item: _Item) -> dict:
        out: dict[str, Any] = {"kind": "Mask"}
        ap = self._aperture(el) if el.aperture is not None else {}
        if ap:
            out["ApertureP"] = ap
        else:
            self.rep.lossy("COLLIMATOR_WITHOUT_APERTURE",
                           "collimator has no aperture; the PALS Mask restricts nothing",
                           element=el.name, kind="Collimator")
        return out

    def _def_marker(self, el, item: _Item) -> dict:
        return {"kind": "Marker"}

    def _def_instrument(self, el: Instrument, item: _Item) -> dict:
        if el.params:
            self.rep.lossy("INSTRUMENT_PARAMS_DROPPED",
                           "diagnostic parameters have no PALS field; only the family survives "
                           "(MetaP.label)", element=el.name, kind="Instrument",
                           params=sorted(el.params))
        return {"kind": "Instrument"}

    def _def_foil(self, el: Foil, item: _Item) -> dict:
        out: dict[str, Any] = {"kind": "Foil"}
        dE = el.meta.get("dE_ref_eV")
        if dE:
            out["FoilP"] = {"dE_ref": _f(dE)}
        if el.material or el.thickness_kg_per_m2:
            self.rep.lossy("FOIL_MATERIAL_DROPPED",
                           "FoilP has only dE_ref; material and thickness have no PALS field",
                           element=el.name, kind="Foil", material=el.material,
                           thickness=el.thickness_kg_per_m2)
        return out

    def _def_taylor(self, el: Taylor, item: _Item) -> dict:
        rows = ("x_out", "px_out", "y_out", "py_out", "z_out", "pz_out")
        group: dict[str, Any] = {}
        for i, key in enumerate(rows):
            terms = []
            if el.offset[i]:
                terms.append(f"term {_f(el.offset[i])!r} 0 0 0 0 0 0")
            for j in range(6):
                c = el.matrix[i][j]
                if c:
                    exps = " ".join("1" if k == j else "0" for k in range(6))
                    terms.append(f"term {_f(c)!r} {exps}")
            if terms:
                group[key] = terms
        return {"kind": "Taylor", "TaylorP": group}

    def _def_patch(self, el: Patch, item: _Item) -> dict:
        group = {"x_offset": _f(el.x_offset), "y_offset": _f(el.y_offset),
                 "z_offset": _f(el.z_offset), "x_rot": _f(el.x_rot), "y_rot": _f(el.y_rot),
                 "z_rot": _f(el.tilt)}
        kind = el.native.get("pals", {}).get("kind")
        if kind == "FloorShift":
            return {"kind": "FloorShift", "CoordinateSetP": group}
        if el.length:
            group["user_sets_length"] = True     # parameters/patch.md: else length is derived
        return {"kind": "Patch", "PatchP": group}

    def _def_referencechange(self, el: ReferenceChange, item: _Item) -> dict:
        group: dict[str, Any] = {}
        if el.energy_eV is not None:
            group["E_tot_ref"] = _f(el.energy_eV + self.lat.reference.species.mass_eV)
        elif el.dE_ref_eV is not None:
            group["dE_ref"] = _f(el.dE_ref_eV)
        if el.dtime_s:
            group["dtime_ref"] = _f(el.dtime_s)
        if el.dphase_rad:
            self.rep.lossy("PALS_REF_PHASE_DROPPED",
                           "ReferenceChangeP has no RF-phase shift; use dtime_ref instead",
                           element=el.name, kind="ReferenceChange", dphase_rad=el.dphase_rad)
        return {"kind": "ReferenceChange", "ReferenceChangeP": group}

    def _def_superposition(self, el: Superposition, item: _Item) -> dict:
        children: dict[str, Any] = {}
        for offset, name in el.children:
            child = self.lat.elements.get(name)
            if child is None:
                self.rep.dropped("SUPERPOSITION_CHILD_MISSING",
                                 f"superposition child {name!r} is not defined",
                                 element=el.name, kind="Superposition")
                continue
            key = self.names.assign(child.name, child)
            node = self._definition(child, key, parent=item)
            if node is None:
                continue
            z = offset + child.length / 2 - el.length / 2
            if abs(z) > _TOL:
                node.setdefault("BodyShiftP", {})["z_offset"] = _f(z)
            children[key] = node
        if el.rf.dE_ref_eV is not None:
            self.rep.lossy("PALS_UNION_DE_REF_DROPPED",
                           "a PALS UnionEle has no RFP group, so the cluster's own dE_ref is "
                           "carried only by its RF children", element=el.name,
                           kind="Superposition", dE_ref=el.rf.dE_ref_eV)
        return {"kind": "UnionEle", "elements": children}

    def _directive(self, el: Directive, rule: Rule) -> dict | None:
        if el.role in _NOTE_ROLES:
            self.notes.append(f"{el.card}: {' '.join(el.args)}".strip(": "))
            self.rep.exact(el.name, "Directive", code="DIRECTIVE_AS_NOTE",
                           message=f"role {el.role!r} carries no physics; kept as a PALS note")
            return None
        if el.role in _MARKER_ROLES:
            self.rep.equivalent("PERIOD_AS_MARKER",
                                f"{el.role} written as a Marker labelled in MetaP; PALS has no "
                                "period construct", element=el.name, kind="Directive")
            return {"kind": "Marker", "MetaP": {"label": el.role}}
        self.notes.append(f"dropped {el.format} directive {el.card} {' '.join(el.args)}".strip())
        self.rep.dropped(rule.code, rule.message, element=el.name, kind="Directive",
                         card=el.card, role=el.role)
        return None


# --------------------------------------------------------------------------- helpers

def _body_shift(shift: BodyShiftP) -> dict:
    """IR ``BodyShiftP.tilt`` is PALS ``z_rot`` (``parameters/bodyshift.md``)."""
    out = {}
    for pals, value in (("x_offset", shift.x_offset), ("y_offset", shift.y_offset),
                        ("z_offset", shift.z_offset), ("x_rot", shift.x_rot),
                        ("y_rot", shift.y_rot), ("z_rot", shift.tilt)):
        if value:
            out[pals] = _f(value)
    return out


def _header(lattice: Lattice, flavor: str) -> str:
    src = lattice.meta.get("source_format", "the lattix IR")
    shape = {"standard": "PALS document",
             "flat": "PALS document (one expanded BeamLine)",
             "beamline": "bare-BeamLine (pre-standard) PALS document"}[flavor]
    return (f"# {shape} written by lattix {__version__} from {src}\n"
            f"# https://github.com/pals-project/pals\n")


def _topo(variables: dict) -> list[str]:
    """Definition order with dependencies first (PALS forbids circular definitions)."""
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


def write(lattice: Lattice, path: str | Path, **options) -> FidelityReport:
    return Writer().write(lattice, Path(path), **options)

