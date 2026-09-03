"""PALS (Particle Accelerator Lattice Standard) reader → lattix IR (PLAN §6 task 2.4).

The standard, read at commit ``a2b1083`` (2026-09-01) of
<https://github.com/pals-project/pals>:

* ``source/fundamentals.md`` — the ``PALS:`` root node (``version``, ``authors``, ``notes``,
  ``reminders``, ``extension_labels``, ``include``, ``load``, ``facility``), the name rules
  (letter or ``_`` first, then alphanumerics and ``_``), the unit system (**SI + eV, angles
  and phases in rad/2π**), the constant/variable/function tables and expressions;
* ``source/lattice-element-kinds.md`` — the 30 element kinds and the parameter groups each
  one may carry;
* ``source/parameters/*.md`` — the groups themselves (see :data:`_GROUPS`);
* ``source/beamlines.md`` — ``BeamLine.line`` items (bare names, in-place definitions,
  ``inherit``, ``repeat`` — **negative means reversed order, not direction reversal** —
  ``direction: -1`` for true reversal, ``placement``, ``zero_point``, ``periodic``);
* ``source/lattice-construction.md`` — ``Lattice.branches``, branch expansion (the first
  element of a branch must be a ``BeginningEle``), ``use:`` (default: the last ``Lattice``),
  ``expand_lattice`` and ``set``;
* ``source/extensions.md`` — extension blocks and ``extension_labels``.

What the names actually are (the IR was designed to mirror PALS, but the standard moved):

===========================  ==================================================
IR                           PALS
===========================  ==================================================
``BendP.angle``              ``BendP.angle_ref``
``BendP.edge_int1`` (fint)   ``BendP.edge1_int`` — the **product** ``fint·hgap`` [m]
``BendP.hgap``               (no separate field; folded into ``edge1_int``)
``BendP.rect``               ``e1_rect``/``e2_rect`` given instead of ``e1``/``e2``
``RFP.n_cell``               ``RFP.num_cells``
``RFP.L_active_m``           ``RFP.L_active`` (defaults to the element ``length``)
``RFP.phase_rad``            ``RFP.phase`` [rad/2π] **plus** ``zero_phase``
``BodyShiftP.tilt``          ``BodyShiftP.z_rot``
``ApertureP.x_limits``       ``ApertureP.x_min``/``x_max`` (or ``x_center``/``x_width``)
``ApertureP.aperture_at``    ``ApertureP.location`` (``ENTRANCE_END`` … ``EVERYWHERE``)
``SolenoidP.Bsol_T``         ``SolenoidP.Bsol`` (or the normalized ``Ksol``)
``Kicker.hkick``/``vkick``   ``MagneticMultipoleP.Kn0L``/``Ks0L`` (see :func:`_kicks_from`)
===========================  ==================================================

Normalized strengths (``Kn{n}``, ``Ks{n}``, ``Ksol``) and a bend given by ``Bn0_ref`` need
the reference momentum, which is only known after the line is expanded, so they are
converted in a second pass over :func:`lattix.ir.walk.propagate` (the pattern the TraceWin
and MAD-X readers use).

Units: PALS is SI + eV like the IR, so only the RF phase (rad/2π → rad, plus the
``zero_phase`` convention) and the normalized ⇄ lab-field rigidity conversions change.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    ApertureP,
    Bend,
    BodyShiftP,
    Collimator,
    Drift,
    Element,
    Foil,
    Instrument,
    Kicker,
    MagneticMultipoleP,
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
    Superposition,
    Taylor,
)
from lattix.ir.expr import Expression, ExpressionError, evaluate
from lattix.ir.lattice import Lattice, Line, LineItem, Variable
from lattix.ir.reference import ReferenceParticle, Species, species
from lattix.ir.units import C_LIGHT, turns_to_rad
from lattix.ir.walk import propagate

# --------------------------------------------------------------------------- constants

#: ``fundamentals.md`` §Constants.  ``pi`` and ``c_light`` also exist in the IR's own table;
#: the rest are added here so a PALS expression can use them.  CODATA-2018 in eV units.
PALS_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "c_light": C_LIGHT,
    "h_planck": 4.135_667_696e-15,          # eV·s
    "hbar": 6.582_119_569e-16,              # eV·s
    "k_boltzmann": 8.617_333_262e-5,        # eV/K
    "r_electron": 2.817_940_3262e-15,       # m
    "r_proton": 1.534_698e-18,             # m
    "e_charge": 1.602_176_634e-19,          # C
    "mu_0": 1.256_637_062e-6,
    "epsilon_0": 8.854_187_8128e-12,
    "classical_radius_factor": 1.439_964_548e-9,   # m·eV
    "fine_structure": 7.297_352_5693e-3,
    "n_avogadro": 6.022_140_76e23,
}

#: PALS parameter groups this reader understands (``lattice-element-parameter-groups.md``).
#: Anything else on an element node is kept verbatim in ``native["pals"]["groups"]``.
_GROUPS = frozenset({
    "ApertureP", "BodyShiftP", "MagneticMultipoleP", "ElectricMultipoleP", "BendP", "RFP",
    "SolenoidP", "MetaP", "TrackingP", "FloorP", "ReferenceP", "ReferenceChangeP", "PatchP",
    "TaylorP", "FoilP", "CoordinateSetP", "TwissP", "ParticleP", "ForkP", "ForkFromP",
    "GirderP", "ACKickerP", "BeamBeamP", "ConverterP",
})

#: Older ``pals-schema`` spellings this reader also accepts (see :data:`_ALIASES`).
#: ``pals-schema`` 0.3.0 (the pals-python reference implementation) is a snapshot of an
#: earlier draft: it has ``SBend``/``RBend`` instead of ``Bend`` and no ``ReferenceChange``.
_KIND_ALIASES = {"SBend": "Bend", "RBend": "Bend"}

#: parameter-group key → the spellings that mean the same thing, newest (the standard text
#: at commit ``a2b1083``) first, then ``pals-schema`` 0.3.0's.
_ALIASES: dict[str, tuple[str, ...]] = {
    "angle_ref": ("angle_ref",),
    "radius_ref": ("radius_ref", "rho_ref"),
    "Bn0_ref": ("Bn0_ref", "bend_field_ref"),
    "edge1_int": ("edge1_int", "edge_int1"),
    "edge2_int": ("edge2_int", "edge_int2"),
    "num_cells": ("num_cells", "n_cell"),
    "dtime_ref": ("dtime_ref", "extra_dtime_ref"),
}

#: PALS kind → IR element class, for the kinds that map one-to-one.
_DIRECT_KINDS: dict[str, type[Element]] = {
    "Drift": Drift,
    "Quadrupole": Quadrupole,
    "Sextupole": Sextupole,
    "Octupole": Octupole,
    "Multipole": Multipole,
    "Bend": Bend,
    "Solenoid": Solenoid,
    "RFCavity": RFCavity,
    "Kicker": Kicker,
    "Marker": Marker,
    "Instrument": Instrument,
    "Foil": Foil,
    "Taylor": Taylor,
    "Patch": Patch,
    "ReferenceChange": ReferenceChange,
    "Mask": Collimator,
    "UnionEle": Superposition,
}

#: Kinds with no IR counterpart (``lattice-element-kinds.md``); a zero-length one becomes a
#: ``Marker`` and a thick one a ``Drift`` so Σlength (invariant I-1) still holds — either way
#: with a DROPPED ``UNSUPPORTED_PALS_KIND`` entry, so ``strict=True`` raises.
_UNSUPPORTED_KINDS = frozenset({
    "ACKicker", "BeamBeam", "CrabCavity", "Converter", "EGun", "Feedback", "Fiducial", "Fork",
    "Girder", "Match", "Placeholder", "Wiggler", "NullEle",
})

#: ``ApertureP.location`` → IR ``ApertureP.aperture_at``.
_APERTURE_AT = {"ENTRANCE_END": "ENTRANCE", "EXIT_END": "EXIT", "BOTH_ENDS": "BOTH_ENDS",
                "EVERYWHERE": "CONTINUOUS"}

#: openPMD ``SpeciesType`` spellings (``fundamentals.md`` §Names) → IR species names.
_SPECIES_ALIASES = {
    "proton": "proton", "anti-proton": "antiproton", "antiproton": "antiproton",
    "electron": "electron", "positron": "positron", "anti-electron": "positron",
    "#1h-1": "h-", "#1h-": "h-", "h-": "h-", "hminus": "h-", "h_minus": "h-",
    "deuteron": "deuteron", "#2h+1": "deuteron",
}

_MAX_INCLUDE_DEPTH = 16


def pals_species_name(sp: Species) -> str:
    """IR species → the openPMD spelling PALS wants in ``ReferenceP.species_ref``."""
    return {"antiproton": "anti-proton", "h-": "#1H-1"}.get(sp.name, sp.name)


def ir_species(name: str) -> Species | None:
    key = str(name).strip().strip('"').lower()
    try:
        return species(_SPECIES_ALIASES.get(key, key))
    except KeyError:
        return None


# --------------------------------------------------------------------------- helpers

def _is_ele_node(v: Any) -> bool:
    return isinstance(v, dict) and "kind" in v


def _one_key(entry: dict) -> tuple[str, Any]:
    (k, v), = entry.items()
    return k, v


def _load_text(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


class _Ctx:
    """Expression environment: PALS constants plus the file's constants and variables."""

    def __init__(self) -> None:
        self.values: dict[str, float] = dict(PALS_CONSTANTS)
        self.variables: dict[str, Variable] = {}

    def define(self, name: str, raw: Any) -> float | None:
        v = self.number(raw)
        if v is not None:
            self.values[name] = v
            self.values[name.lower()] = v
            self.variables[name] = Variable(
                value=v, expression=Expression(text=str(raw)) if isinstance(raw, str) else None)
        return v

    def number(self, raw: Any) -> float | None:
        """A PALS real: a number, ``null``, ``Inf``/``-Inf``, or an expression string."""
        if raw is None:
            return None
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        if isinstance(raw, (int, float)):
            return float(raw)
        text = str(raw).strip()
        if not text or text.lower() in ("null", "none"):
            return None
        if text in ("Inf", "+Inf"):
            return math.inf
        if text == "-Inf":
            return -math.inf
        return evaluate(text, self.values)          # raises ExpressionError


# --------------------------------------------------------------------------- the reader

class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "pals"

    def read(self, path: Path, *, use: str | None = None, strict: bool = False,
             species: str | Species | None = None, kinetic_energy_eV: float | None = None,
             **_ignored) -> tuple[Lattice, FidelityReport]:
        """Parse a PALS YAML (or JSON) document.

        ``use`` overrides the document's own ``use:`` statement and may name a ``Lattice``,
        a ``BeamLine`` or a branch.  ``species``/``kinetic_energy_eV`` supply the reference
        particle when the document has no ``BeginningEle``/``ReferenceP`` (recorded as
        ``EQUIVALENT:PALS_REFERENCE_ASSUMED``).  In strict mode the first LOSSY/DROPPED
        entry raises :class:`~lattix.fidelity.TranslationError`.
        """
        path = Path(path).resolve()
        self.rep = FidelityReport(source_format="pals", source_file=str(path))
        self.path = path
        self.ctx = _Ctx()
        self.pending: list[tuple[Element, Any]] = []
        self.warnings: list[str] = []
        self.meta: dict = {}

        raw = _load_text(path)
        root = self._pals_root(raw)
        root = self._apply_load(root, path.parent, seen={path})
        root = self._resolve_includes(root, path.parent, depth=0)

        for key in ("version", "notes", "reminders", "authors", "extension_labels",
                    "phase_space_coordinates"):
            if root.get(key) is not None:
                self.meta[f"pals_{key}"] = root[key]

        facility = root.get("facility") or []
        if not isinstance(facility, list):
            facility = [facility]

        self.elements: dict[str, dict] = {}      # name -> raw node (kind is an element kind)
        self.beamlines: dict[str, dict] = {}
        self.lattices: dict[str, dict] = {}
        self.doc_use: str | None = None
        self._scan(facility)

        lat = Lattice(name=path.stem.replace(".pals", ""), reference=self._placeholder_reference())
        lat.variables.update(self.ctx.variables)
        self.lat = lat
        self.built: dict[str, Element] = {}
        self.line_cache: dict[str, str] = {}
        self.reference: ReferenceParticle | None = None

        root_line = self._root_line_name(use)
        if root_line is not None:
            lat.use = self._build_line(root_line, stack=())
        else:
            lat.use = self._salvage_line()
        lat.name = lat.use or lat.name

        lat.reference = self._reference(species, kinetic_energy_eV)
        lat.meta.update(self.meta)
        lat.warnings.extend(self.warnings)
        self._second_pass()
        self.rep.raise_if(strict)
        return lat, self.rep

    # -- document assembly --------------------------------------------------
    def _pals_root(self, raw: Any) -> dict:
        """The ``PALS:`` node, or a bare document (ImpactX writes a one-key ``BeamLine``)."""
        if isinstance(raw, dict) and "PALS" in raw:
            node = raw["PALS"]
            return dict(node) if isinstance(node, dict) else {}
        if isinstance(raw, dict):
            self.rep.equivalent("PALS_ROOT_MISSING",
                                "document has no 'PALS:' root node; its top level is read as a "
                                "facility list (the shape ImpactX's KnownElementsList writes)")
            return {"facility": [{k: v} for k, v in raw.items()]}
        self.rep.dropped("PALS_NOT_A_MAPPING", f"{type(raw).__name__} document is not a PALS file")
        return {}

    def _resolve_includes(self, node: Any, base: Path, depth: int) -> Any:
        """``fundamentals.md`` §Including Other Files: verbatim insert at the include point."""
        if depth > _MAX_INCLUDE_DEPTH:
            self.rep.lossy("INCLUDE_NOT_FOLLOWED",
                           f"include nesting deeper than {_MAX_INCLUDE_DEPTH}; stopped")
            return node
        if isinstance(node, list):
            out: list[Any] = []
            for item in node:
                if isinstance(item, dict) and set(item) == {"include"}:
                    got = self._read_include(item["include"], base, depth)
                    if isinstance(got, list):
                        out.extend(got)
                    elif isinstance(got, dict):
                        out.extend({k: v} for k, v in got.items())
                    continue
                out.append(self._resolve_includes(item, base, depth))
            return out
        if isinstance(node, dict):
            out_d: dict[str, Any] = {}
            for k, v in node.items():
                if k == "include":
                    got = self._read_include(v, base, depth)
                    if isinstance(got, dict):
                        out_d.update(got)
                    elif got is not None:
                        self.rep.lossy("INCLUDE_NOT_FOLLOWED",
                                       f"include {v!r} is not a mapping; ignored at this level")
                    continue
                out_d[k] = self._resolve_includes(v, base, depth)
            return out_d
        return node

    def _read_include(self, spec: Any, base: Path, depth: int) -> Any:
        target = (base / str(spec)).resolve()
        if not target.is_file():
            self.rep.lossy("INCLUDE_NOT_FOLLOWED", f"included file {spec!r} not found",
                           file=str(target))
            return None
        got = _load_text(target)
        return self._resolve_includes(got, target.parent, depth + 1)

    def _apply_load(self, root: dict, base: Path, seen: set[Path]) -> dict:
        """``fundamentals.md`` §Load Files: merge whole ``PALS`` nodes, honouring ``SELF``."""
        names = root.get("load")
        if not names:
            return root
        me = {k: v for k, v in root.items() if k != "load"}
        parts: list[dict] = []
        has_self = False
        for name in (names if isinstance(names, list) else [names]):
            if str(name).strip() == "SELF":
                parts.append(me)
                has_self = True
                continue
            target = (base / str(name)).resolve()
            if not target.is_file() or target in seen:
                self.rep.lossy("INCLUDE_NOT_FOLLOWED",
                               f"loaded file {name!r} {'is circular' if target in seen else 'not found'}",
                               file=str(target))
                continue
            sub = self._pals_root(_load_text(target))
            sub = self._apply_load(sub, target.parent, seen | {target})
            parts.append(self._resolve_includes(sub, target.parent, depth=0))
        if not has_self:
            parts.append(me)
        return _merge_pals(parts)

    # -- facility scan ------------------------------------------------------
    def _scan(self, facility: list) -> None:
        for entry in facility:
            if isinstance(entry, str):
                if entry.strip() != "expand_lattice":
                    self.rep.lossy("PALS_UNKNOWN_FACILITY_ENTRY", f"ignored facility item {entry!r}")
                continue
            if not isinstance(entry, dict) or len(entry) != 1:
                self.rep.lossy("PALS_UNKNOWN_FACILITY_ENTRY",
                               f"facility items must be one-key mappings, got {entry!r:.60}")
                continue
            key, val = _one_key(entry)
            if key == "use":
                self.doc_use = str(val)
            elif key == "expand_lattice":
                continue
            elif key in ("constants", "variables"):
                for item in (val if isinstance(val, list) else [val]):
                    if isinstance(item, dict):
                        for n, raw in item.items():
                            self._define(n, raw)
            elif key in ("set", "sets", "superimpose"):
                self.rep.lossy("PALS_COMMAND_IGNORED",
                               f"'{key}' command is not applied; parameters keep their "
                               "definition-time values", command=key)
            elif _is_ele_node(val):
                kind = str(val["kind"])
                if kind == "BeamLine":
                    self.beamlines[key] = val
                elif kind == "Lattice":
                    self.lattices[key] = val
                elif kind in ("constant", "variable"):
                    self._define(key, val.get("value"))
                else:
                    self.elements[key] = val
            else:
                self.meta.setdefault("pals_extensions", {})[key] = val

    def _define(self, name: str, raw: Any) -> None:
        try:
            self.ctx.define(name, raw)
        except ExpressionError as exc:
            self.rep.lossy("PALS_EXPRESSION_UNEVALUABLE",
                           f"constant {name!r}: {exc}", element=None)

    # -- root selection -----------------------------------------------------
    def _root_line_name(self, use: str | None) -> str | None:
        """The BeamLine to expand: ``use=`` → a Lattice/branch/BeamLine, else the document's
        ``use:``, else the last ``Lattice``'s first branch (``lattice-construction.md`` §use)."""
        wanted = use or self.doc_use
        if wanted is not None:
            if wanted in self.beamlines:
                return wanted
            if wanted in self.lattices:
                return self._first_branch(wanted)
            for lat_name, node in self.lattices.items():
                for branch, root in self._branches(node):
                    if branch == wanted:
                        self.meta["pals_lattice"] = lat_name
                        self.meta["pals_branch"] = branch
                        return root
            self.rep.lossy("PALS_USE_NOT_FOUND",
                           f"use={wanted!r} names no Lattice, branch or BeamLine")
        if self.lattices:
            return self._first_branch(list(self.lattices)[-1])
        if self.beamlines:
            return list(self.beamlines)[-1]
        return None

    def _salvage_line(self) -> str:
        """No ``BeamLine`` anywhere — a settings/definitions fragment.  Place the element
        definitions in facility order so the lattice is still usable, and say so."""
        name = "pals_facility"
        line = Line(name=name)
        for el_name, node in self.elements.items():
            if self._begin_node(el_name, node):
                continue
            line.items.append(LineItem(ref=self._element(el_name, node).name))
        self.lat.lines[name] = line
        self.rep.dropped("PALS_NO_LINE",
                         f"the document defines no BeamLine; its {len(line.items)} element "
                         "definition(s) are placed in facility order",
                         elements=len(line.items))
        return name

    def _branches(self, node: dict) -> list[tuple[str, str]]:
        """``[(branch name, root BeamLine name)]`` of a ``Lattice`` node."""
        raw = node.get("branches") or []
        if isinstance(raw, (str, dict)):
            raw = [raw]
        out: list[tuple[str, str]] = []
        for item in raw:
            if isinstance(item, str):
                out.append((item, item))
            elif isinstance(item, dict) and len(item) == 1:
                name, body = _one_key(item)
                root = str(body.get("inherit", name)) if isinstance(body, dict) else name
                out.append((name, root))
        return out

    def _first_branch(self, lattice_name: str) -> str | None:
        branches = self._branches(self.lattices[lattice_name])
        if not branches:
            self.rep.dropped("PALS_NO_LINE", f"Lattice {lattice_name!r} has no branches")
            return None
        if len(branches) > 1:
            self.rep.lossy("PALS_EXTRA_BRANCHES",
                           f"Lattice {lattice_name!r} has {len(branches)} branches; only "
                           f"{branches[0][0]!r} is read (pass use= for another)",
                           branches=[b for b, _ in branches])
        self.meta["pals_lattice"] = lattice_name
        self.meta["pals_branch"] = branches[0][0]
        return branches[0][1]

    # -- lines --------------------------------------------------------------
    def _build_line(self, name: str, stack: tuple[str, ...]) -> str:
        """Register the IR :class:`~lattix.ir.lattice.Line` for BeamLine *name*; returns its key."""
        if name in self.line_cache:
            return self.line_cache[name]
        if name in stack:
            self.rep.dropped("PALS_LINE_RECURSION", f"BeamLine {name!r} contains itself")
            self.lat.lines[name] = Line(name=name)
            self.line_cache[name] = name
            return name
        node = self.beamlines[name]
        if node.get("periodic"):
            self.meta.setdefault("pals_periodic", []).append(name)
        if node.get("multipass"):
            self.rep.lossy("PALS_MULTIPASS_LINE", f"BeamLine {name!r} is a multipass line; the IR "
                           "has no multipass construct", element=name)
        if node.get("zero_point"):
            self.rep.lossy("PALS_ZERO_POINT_IGNORED",
                           f"BeamLine {name!r} zero_point is only used by placement", element=name)
        line = Line(name=name)
        self.lat.lines[name] = line
        self.line_cache[name] = name
        for index, item in enumerate(node.get("line") or []):
            it = self._line_item(item, name, index, stack + (name,))
            if it is not None:
                line.items.append(it)
        return name

    def _line_item(self, item: Any, line_name: str, index: int,
                   stack: tuple[str, ...]) -> LineItem | None:
        if isinstance(item, str):
            return self._reference_item(item, {}, stack)
        if not isinstance(item, dict) or len(item) != 1:
            self.rep.lossy("PALS_UNKNOWN_LINE_ITEM",
                           f"line {line_name!r} item {index} is not a name or a one-key mapping")
            return None
        name, body = _one_key(item)
        body = dict(body) if isinstance(body, dict) else {}
        if body.get("placement") is not None:
            self.rep.lossy("PALS_PLACEMENT_IGNORED",
                           f"placement of {name!r} in {line_name!r} ignored; items are laid out "
                           "end to end", element=name)
            body.pop("placement")
        parent = body.pop("inherit", None)
        kind = body.get("kind")
        if kind == "BeamLine" or (parent is None and kind is None and name in self.beamlines):
            if kind == "BeamLine":
                self.beamlines.setdefault(name, item[name])
            return self._reference_item(name, body, stack)
        if kind is not None or parent is not None or any(k in body for k in _GROUPS) or "length" in body:
            # an in-place definition, an `inherit:`, or a per-occurrence override
            base = str(parent) if parent is not None else (name if name in self.elements else None)
            merged = _inherit(self.elements.get(base, {}) if base else {}, body)
            if "kind" not in merged and base is None:
                self.rep.lossy("PALS_UNKNOWN_LINE_ITEM",
                               f"line item {name!r} has parameters but no kind and no inherit")
                return None
            if self._begin_node(name, merged):
                return None
            el = self._element(name, merged, unique=name in self.built)
            mods, _rev = _modifiers(body)
            return LineItem(ref=el.name, **mods)
        return self._reference_item(name, body, stack)

    def _reference_item(self, name: str, body: dict, stack: tuple[str, ...]) -> LineItem | None:
        mods, reversed_order = _modifiers(body)
        if name in self.beamlines:
            ref = self._build_line(name, stack)
            if reversed_order:
                ref = self._reversed_line(ref, stack)
            return LineItem(ref=ref, **mods)
        if name in self.elements:
            if self._begin_node(name, self.elements[name]):
                return None
            return LineItem(ref=self._element(name, self.elements[name]).name, **mods)
        self.rep.dropped("PALS_UNDEFINED_ITEM", f"line item {name!r} is not defined", element=name)
        return None

    def _reversed_line(self, name: str, stack: tuple[str, ...]) -> str:
        """``repeat: -N`` reverses the *order* only (``beamlines.md`` §Repetition), which the IR's
        ``LineItem.reverse`` (a true direction reversal) does not mean — so materialise an
        explicitly reversed copy of the line instead.  Physics-exact; only the name is new."""
        rev = f"{name}__reversed"
        if rev in self.lat.lines:
            return rev
        src = self.lat.lines[name]
        out = Line(name=rev)
        self.lat.lines[rev] = out
        for it in reversed(src.items):
            ref = it.ref
            if ref in self.lat.lines and not it.reverse:
                ref = self._reversed_line(ref, stack)
            out.items.append(LineItem(ref=ref, repeat=it.repeat, reverse=it.reverse))
        self.rep.equivalent("PALS_REVERSED_ORDER_EXPANDED",
                            f"'repeat: -N' on {name!r} reverses element order without direction "
                            f"reversal; expanded into the explicit line {rev!r}", element=name)
        return rev

    # -- elements -----------------------------------------------------------
    def _element(self, name: str, node: dict, *, unique: bool = False) -> Element:
        key = name
        if not unique and key in self.built:
            return self.built[key]
        node = _inherit(self.elements.get(str(node["inherit"]), {}), node) if "inherit" in node else node
        kind = _KIND_ALIASES.get(str(node.get("kind", "Marker")), str(node.get("kind", "Marker")))
        el = self._make(name, kind, node)
        if kind in _DIRECT_KINDS:
            self.rep.exact(name, el.kind)
        el.provenance = Provenance(format="pals", file=str(self.path), original_name=name,
                                   original_type=kind)
        el.name = self.lat.add_element(el)
        if not unique:
            self.built[key] = el
        return el

    def _make(self, name: str, kind: str, node: dict) -> Element:
        groups = {k: (dict(v) if isinstance(v, dict) else v) for k, v in node.items()
                  if k in _GROUPS}
        extra = {k: v for k, v in node.items()
                 if k not in _GROUPS and k not in ("kind", "length", "inherit", "repeat",
                                                   "direction", "placement", "multipass",
                                                   "zero_point", "periodic", "line", "elements")}
        length = self._num(self._get(node, "length", 0.0), name, "length") or 0.0

        cls = _DIRECT_KINDS.get(kind)
        if cls is None:
            el = self._unsupported(name, kind, length)
        elif kind == "UnionEle":
            el = self._union(name, node, length)
        else:
            el = cls(name=name, length=length)
        if isinstance(node.get("length"), str):
            el.expressions["length"] = Expression(text=node["length"])

        self._apply_groups(el, kind, groups)
        keep = {k: v for k, v in groups.items() if k in _PASSTHROUGH_GROUPS}
        if keep or extra:
            el.native.setdefault("pals", {})["groups"] = {**keep, **extra}
        if kind in ("Mask", "FloorShift"):
            el.native.setdefault("pals", {})["kind"] = kind
        return el

    def _unsupported(self, name: str, kind: str, length: float) -> Element:
        if kind == "FloorShift":
            self.rep.equivalent("PALS_FLOORSHIFT_AS_PATCH",
                                "FloorShift shifts the floor without affecting tracking; the IR "
                                "Patch shifts both", element=name, kind="Patch")
            return Patch(name=name, length=length)
        replacement = "Drift" if length else "Marker"
        known = "" if kind in _UNSUPPORTED_KINDS else " (and is not a PALS kind either)"
        self.rep.dropped("UNSUPPORTED_PALS_KIND",
                         f"PALS kind {kind!r} has no IR counterpart{known}; read as a "
                         f"{replacement.lower()} of the same length", element=name, kind=kind,
                         pals_kind=kind, replacement=replacement, length=length)
        return (Drift if length else Marker)(name=name, length=length)

    def _union(self, name: str, node: dict, length: float) -> Superposition:
        """``UnionEle.elements`` are centred on the union's centre and shifted by
        ``BodyShiftP.z_offset`` (``lattice-element-kinds.md`` §UnionEle)."""
        el = Superposition(name=name, length=length)
        for child_name, child in (node.get("elements") or {}).items():
            if not isinstance(child, dict):
                continue
            child = dict(child)
            shift = dict(child.get("BodyShiftP") or {})
            z = self._num(shift.pop("z_offset", 0.0), child_name, "z_offset") or 0.0
            if shift:
                child["BodyShiftP"] = shift
            else:
                child.pop("BodyShiftP", None)
            kid = self._element(child_name, child, unique=child_name in self.built)
            el.children.append((z + length / 2 - kid.length / 2, kid.name))
        self.rep.equivalent("PALS_UNIONELE_AS_SUPERPOSITION",
                            f"UnionEle {name!r} read as a Superposition of {len(el.children)} "
                            "children at their z offsets from the union entrance", element=name,
                            kind="Superposition")
        return el

    # -- parameter groups ---------------------------------------------------
    def _apply_groups(self, el: Element, kind: str, groups: dict) -> None:
        if "ApertureP" in groups:
            self._aperture(el, groups["ApertureP"])
        if "BodyShiftP" in groups:
            self._body_shift(el, groups["BodyShiftP"])
        if "TrackingP" in groups:
            el.tracking.update(groups["TrackingP"] or {})
        if "MetaP" in groups:
            el.meta["MetaP"] = groups["MetaP"]
            alias = (groups["MetaP"] or {}).get("alias")
            if alias and el.provenance is None:
                el.meta["pals_alias"] = alias
        for name in ("FloorP", "ReferenceP", "TwissP"):
            if name in groups:
                el.meta[name] = groups[name]
        if "ElectricMultipoleP" in groups and any((groups["ElectricMultipoleP"] or {}).values()):
            self.rep.lossy("PALS_ELECTRIC_MULTIPOLE_DROPPED",
                           "ElectricMultipoleP has no IR counterpart", element=el.name, kind=el.kind)
        if isinstance(el, Bend):
            self._bend(el, groups.get("BendP") or {})
        if isinstance(el, RFCavity):
            self._rf(el, groups.get("RFP") or {}, groups.get("SolenoidP") or {})
        elif isinstance(el, Solenoid):
            self._solenoid(el, groups.get("SolenoidP") or {})
        if "MagneticMultipoleP" in groups:
            self._multipole(el, groups["MagneticMultipoleP"] or {})
        if isinstance(el, ReferenceChange):
            self._reference_change(el, groups.get("ReferenceChangeP") or {})
        if isinstance(el, Patch):
            self._patch(el, groups.get("PatchP") or groups.get("CoordinateSetP") or {})
        if isinstance(el, Foil) and "FoilP" in groups:
            dE = self._num((groups["FoilP"] or {}).get("dE_ref"), el.name, "dE_ref")
            if dE:
                el.meta["dE_ref_eV"] = dE
                self.rep.lossy("PALS_FOIL_DE_REF_DROPPED",
                               "FoilP.dE_ref has no IR field; kept in meta only",
                               element=el.name, kind="Foil", dE_ref=dE)
        if isinstance(el, Taylor) and "TaylorP" in groups:
            self._taylor(el, groups["TaylorP"] or {})
        if isinstance(el, Instrument):
            label = (groups.get("MetaP") or {}).get("label")
            if label:
                el.family = str(label)

    def _num(self, raw: Any, element: str, what: str) -> float | None:
        try:
            return self.ctx.number(raw)
        except ExpressionError as exc:
            self.rep.lossy("PALS_EXPRESSION_UNEVALUABLE",
                           f"{what} = {raw!r}: {exc}; treated as unset", element=element, what=what)
            return None

    @staticmethod
    def _get(node: dict, key: str, default: Any = None) -> Any:
        v = node.get(key, default)
        return default if v is None else v

    @staticmethod
    def _alias(group: dict, key: str) -> Any:
        """The value of *key* under any spelling the standard or pals-schema uses."""
        for name in _ALIASES.get(key, (key,)):
            if group.get(name) is not None:
                return group[name]
        return None

    def _aperture(self, el: Element, g: dict) -> None:
        shape = str(g.get("shape", "ELLIPTICAL")).upper()
        if shape not in ("ELLIPTICAL", "RECTANGULAR"):
            self.rep.lossy("PALS_APERTURE_SHAPE_UNSUPPORTED",
                           f"aperture shape {shape!r} has no IR counterpart; kept in native only",
                           element=el.name, kind=el.kind, shape=shape)
            return
        if g.get("aperture_active") is False:
            self.rep.equivalent("PALS_APERTURE_INACTIVE",
                                "aperture_active is false; no IR aperture is created",
                                element=el.name, kind=el.kind)
            return
        limits = {}
        for axis in ("x", "y"):
            pair = g.get(f"{axis}_limits")          # pals-schema 0.3.0 spelling
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                lo = self._num(pair[0], el.name, f"{axis}_limits")
                hi = self._num(pair[1], el.name, f"{axis}_limits")
            else:
                lo = self._num(g.get(f"{axis}_min"), el.name, f"{axis}_min")
                hi = self._num(g.get(f"{axis}_max"), el.name, f"{axis}_max")
            if lo is None and hi is None:
                c = self._num(g.get(f"{axis}_center"), el.name, f"{axis}_center")
                w = self._num(g.get(f"{axis}_width"), el.name, f"{axis}_width")
                if w is not None:
                    c = c or 0.0
                    lo, hi = c - w / 2, c + w / 2
            limits[axis] = None if (lo is None or hi is None) else (lo, hi)
        if limits["x"] is None and limits["y"] is None:
            return
        loc = str(g.get("location", "ENTRANCE_END")).upper()
        at = _APERTURE_AT.get(loc)
        if at is None:
            self.rep.lossy("PALS_APERTURE_LOCATION",
                           f"aperture location {loc!r} has no IR counterpart; read as BOTH_ENDS",
                           element=el.name, kind=el.kind, location=loc)
            at = "BOTH_ENDS"
        el.aperture = ApertureP(shape=shape, x_limits=limits["x"], y_limits=limits["y"],
                                aperture_at=at)

    def _body_shift(self, el: Element, g: dict) -> None:
        vals = {ir: self._num(g.get(p), el.name, p) or 0.0
                for ir, p in (("x_offset", "x_offset"), ("y_offset", "y_offset"),
                              ("z_offset", "z_offset"), ("x_rot", "x_rot"),
                              ("y_rot", "y_rot"), ("tilt", "z_rot"))}
        shift = BodyShiftP(**vals)
        if not shift.is_zero():
            el.shift = shift

    def _multipole(self, el: Element, g: dict) -> None:
        mp: MagneticMultipoleP | None = getattr(el, "multipole", None)
        kicks: dict[str, float] = {}
        pending: list[tuple[str, int, float, bool]] = []
        forms: dict[str, str] = {}
        for key, raw in (g or {}).items():
            parsed = _parse_multipole_key(str(key))
            if parsed is None:
                if str(key).endswith("_taper"):
                    self.rep.lossy("PALS_MULTIPOLE_TAPER_DROPPED",
                                   f"tapering parameter {key!r} has no IR field",
                                   element=el.name, kind=el.kind)
                else:
                    self.rep.lossy("PALS_UNKNOWN_PARAMETER",
                                   f"MagneticMultipoleP.{key} is not a PALS multipole parameter",
                                   element=el.name, kind=el.kind)
                continue
            what, order, integrated = parsed
            value = self._num(raw, el.name, str(key))
            if value is None:
                continue
            if what != "tilt":
                forms["ns"[what in ("Bs", "Ks")] + str(order)] = str(key)[0]
            if isinstance(el, Kicker) and order == 0:
                if what == "tilt":
                    self.rep.lossy("PALS_KICKER_TILT_DROPPED",
                                   "tilt0 on a Kicker has no IR field; give Bn0/Bs0 instead",
                                   element=el.name, kind="Kicker")
                else:
                    kicks[str(key)] = value
                continue
            if mp is None:
                self.rep.lossy("PALS_MULTIPOLE_DROPPED",
                               f"{el.kind} carries no IR multipole group; {key} dropped",
                               element=el.name, kind=el.kind)
                continue
            if what == "tilt":
                mp.tilt[order] = value
            elif what in ("Bn", "Bs"):
                self._store_field(el, mp, what, order, value, integrated)
            else:                                    # Kn / Ks: needs the reference momentum
                pending.append((what, order, value, integrated))
        if pending:
            self.pending.append((el, ("normalized", pending)))
        if kicks:
            self.pending.append((el, ("kick", kicks)))
        if forms:
            # {"n1": "K"} -> order 1 normal was written normalized; the writer emits it back
            # in the same form (native[fmt] is re-emitted only to the same format).
            el.native.setdefault("pals", {}).setdefault("multipole_form", {}).update(forms)

    def _store_field(self, el: Element, mp: MagneticMultipoleP, what: str, order: int,
                     value: float, integrated: bool) -> None:
        table = (mp.BnL if what == "Bn" else mp.BsL) if integrated else (mp.Bn if what == "Bn" else mp.Bs)
        if isinstance(el, Multipole):
            # the IR Multipole is a thin element carrying integrated strengths
            target = mp.BnL if what == "Bn" else mp.BsL
            if integrated:
                target[order] = value
            else:
                target[order] = value * el.length
                self.rep.equivalent("PALS_MULTIPOLE_INTEGRATED",
                                    f"{what}{order} = {value:g} over {el.length:g} m read as the "
                                    f"integrated {what}{order}L", element=el.name, kind="Multipole")
            return
        if integrated and el.length:
            (mp.Bn if what == "Bn" else mp.Bs)[order] = value / el.length
            self.rep.equivalent("PALS_INTEGRATED_TO_FIELD",
                                f"{what}{order}L = {value:g} T·m^(1-n) over {el.length:g} m read as "
                                f"{what}{order} = {value / el.length:g}", element=el.name, kind=el.kind)
            return
        table[order] = value

    def _solenoid(self, el: Solenoid, g: dict) -> None:
        bsol = self._num(g.get("Bsol"), el.name, "Bsol")
        if bsol is not None:
            el.solenoid.Bsol_T = bsol
            return
        ksol = self._num(g.get("Ksol"), el.name, "Ksol")
        if ksol is not None:
            el.native.setdefault("pals", {})["solenoid_form"] = "Ksol"
            self.pending.append((el, ("ksol", ksol)))

    def _bend(self, el: Bend, g: dict) -> None:
        b = el.bend
        b.tilt_ref = self._num(g.get("tilt_ref"), el.name, "tilt_ref") or 0.0
        b.edge_int1 = self._num(self._alias(g, "edge1_int"), el.name, "edge1_int") or 0.0
        edge2 = self._num(self._alias(g, "edge2_int"), el.name, "edge2_int")
        b.edge_int2 = edge2
        if b.edge_int1 or edge2:
            # PALS stores only the product fint·hgap [m] (parameters/bend.md §edge1_int);
            # hgap = 1 m keeps the product exact and is undone on write.
            b.hgap = 1.0
            self.rep.equivalent("PALS_EDGE_INT_PRODUCT",
                                "PALS edge1_int/edge2_int are the products fint·hgap; read with "
                                "hgap = 1 m so the product is preserved exactly",
                                element=el.name, kind="Bend", edge1_int=b.edge_int1,
                                edge2_int=edge2)
        for key in ("h1", "h2"):
            if self._num(g.get(key), el.name, key):
                self.rep.lossy("PALS_POLE_FACE_CURVATURE_DROPPED",
                               f"BendP.{key} (pole-face curvature) has no IR field",
                               element=el.name, kind="Bend")
        for key, default in (("ref_geometry", "ARC"), ("multipole_geometry", "FOLLOWS_REF_GEOMETRY")):
            got = str(g.get(key, default)).upper()
            if got != default:
                self.rep.lossy("PALS_BEND_GEOMETRY_IGNORED",
                               f"BendP.{key} = {got!r} is treated as {default!r}",
                               element=el.name, kind="Bend", **{key: got})
        self._bend_shape(el, g)
        self._bend_edges(el, g)

    def _bend_shape(self, el: Bend, g: dict) -> None:
        """Resolve two of {curvature set, length set, ``angle_ref``} (``parameters/bend.md``)."""
        angle = self._num(g.get("angle_ref"), el.name, "angle_ref")
        radius = self._num(self._alias(g, "radius_ref"), el.name, "radius_ref")
        gref = self._num(g.get("g_ref"), el.name, "g_ref")
        if gref and radius is None:
            radius = 1.0 / gref
        chord = self._num(g.get("L_chord"), el.name, "L_chord")
        rect_len = self._num(g.get("L_rectangle"), el.name, "L_rectangle")
        form = [k for k in ("angle_ref", "radius_ref", "rho_ref", "g_ref", "L_chord",
                            "L_rectangle") if g.get(k) is not None]
        if form:
            el.native.setdefault("pals", {})["bend_form"] = form

        if angle is not None and el.length:
            pass
        elif angle is not None and radius is not None:
            el.length = angle * radius
        elif angle is not None and chord is not None:
            el.length = chord if angle == 0 else chord * (angle / 2) / math.sin(angle / 2)
        elif angle is not None and rect_len is not None:
            el.length = rect_len if angle == 0 else rect_len * angle / math.sin(angle)
        elif angle is None and radius is not None and el.length:
            angle = el.length / radius
        elif angle is None and (bn0 := self._num(self._alias(g, "Bn0_ref"),
                                                  el.name, "Bn0_ref")) is not None:
            self.pending.append((el, ("bend_field", bn0)))
            angle = 0.0
        el.bend.angle = angle or 0.0
        if isinstance(g.get("angle_ref"), str):
            el.expressions["bend.angle"] = Expression(text=g["angle_ref"])

    def _bend_edges(self, el: Bend, g: dict) -> None:
        """``e1 = e1_rect + angle_ref/2`` for ``ref_geometry: ARC`` (``parameters/bend.md``)."""
        half = el.bend.angle / 2.0
        e1, e2 = self._num(g.get("e1"), el.name, "e1"), self._num(g.get("e2"), el.name, "e2")
        e1r = self._num(g.get("e1_rect"), el.name, "e1_rect")
        e2r = self._num(g.get("e2_rect"), el.name, "e2_rect")
        if e1 is None and e1r is not None:
            e1, el.bend.rect = e1r + half, True
        if e2 is None and e2r is not None:
            e2, el.bend.rect = e2r + half, True
        el.bend.e1 = e1 or 0.0
        el.bend.e2 = e2 or 0.0

    def _rf(self, el: RFCavity, g: dict, sol: dict) -> None:
        rf = el.rf
        rf.frequency_Hz = self._num(g.get("frequency"), el.name, "frequency")
        rf.L_active_m = self._num(g.get("L_active"), el.name, "L_active")
        n = self._num(self._alias(g, "num_cells"), el.name, "num_cells")
        rf.n_cell = int(n) if n else None
        rf.dE_ref_eV = self._num(g.get("dE_ref"), el.name, "dE_ref")
        ctype = str(g.get("cavity_type", "STANDING_WAVE")).upper()
        if ctype in ("STANDING_WAVE", "TRAVELING_WAVE"):
            rf.cavity_type = ctype
        active = rf.L_active_m if rf.L_active_m is not None else el.length
        volt = self._num(g.get("voltage"), el.name, "voltage")
        grad = self._num(g.get("gradient"), el.name, "gradient")
        if volt is None and grad is not None:
            volt = grad * active                       # parameters/rf.md: voltage = gradient·L_active
            el.native.setdefault("pals", {})["rf_form"] = "gradient"
        elif volt is not None and grad is None and active:
            grad = volt / active
        rf.voltage_V = volt or 0.0
        rf.gradient_V_per_m = grad
        phase = turns_to_rad(self._num(g.get("phase"), el.name, "phase") or 0.0)
        zero = str(g.get("zero_phase", "ACCELERATING")).upper()
        if zero in ("BELOW_TRANSITION", "ABOVE_TRANSITION"):
            # phase = 0 is the stable zero crossing there; the IR's 0 is the crest, and the
            # stable crossing sits a quarter period before (below) / after (above) transition.
            phase += -math.pi / 2 if zero == "BELOW_TRANSITION" else math.pi / 2
            self.rep.equivalent("PALS_ZERO_PHASE_TRANSITION",
                                f"RFP.zero_phase = {zero}; converted to the IR crest convention "
                                f"(phase {'-' if zero == 'BELOW_TRANSITION' else '+'} pi/2)",
                                element=el.name, kind="RFCavity", zero_phase=zero)
        rf.phase_rad = phase
        if g.get("harmon") is not None and rf.frequency_Hz is None:
            self.rep.lossy("PALS_HARMON_UNRESOLVED",
                           "RFP.harmon needs the branch 1-turn time to become a frequency; "
                           "the IR has no harmonic-number field",
                           element=el.name, kind="RFCavity", harmon=g["harmon"])
        if g.get("multipass_phase"):
            self.rep.lossy("PALS_MULTIPASS_PHASE_DROPPED", "RFP.multipass_phase has no IR field",
                           element=el.name, kind="RFCavity")
        if sol.get("Bsol") or sol.get("Ksol"):
            self.rep.lossy("PALS_RF_SOLENOID_DROPPED",
                           "an RFCavity's DC SolenoidP field has no IR field on RFCavity",
                           element=el.name, kind="RFCavity")

    def _reference_change(self, el: ReferenceChange, g: dict) -> None:
        el.dE_ref_eV = self._num(g.get("dE_ref"), el.name, "dE_ref")
        el.dtime_s = self._num(self._alias(g, "dtime_ref"), el.name, "dtime_ref")
        etot = self._num(g.get("E_tot_ref"), el.name, "E_tot_ref")
        if etot is not None:
            el.meta["pals_E_tot_ref"] = etot         # kinetic energy needs the species: second pass
            self.pending.append((el, ("refchange_total", etot)))
        for key in ("dpc_ref", "pc_ref", "species_ref", "time_ref"):
            if g.get(key) is not None:
                self.rep.lossy("PALS_REFERENCE_CHANGE_DROPPED",
                               f"ReferenceChangeP.{key} has no IR field", element=el.name,
                               kind="ReferenceChange", parameter=key)

    def _patch(self, el: Patch, g: dict) -> None:
        for ir, key in (("x_offset", "x_offset"), ("y_offset", "y_offset"),
                        ("z_offset", "z_offset"), ("x_rot", "x_rot"), ("y_rot", "y_rot"),
                        ("tilt", "z_rot")):
            setattr(el, ir, self._num(g.get(key), el.name, key) or 0.0)
        if g.get("flexible"):
            self.rep.lossy("PALS_FLEXIBLE_PATCH",
                           "a flexible Patch takes its geometry from the next element; read as a "
                           "rigid patch with the given offsets", element=el.name, kind="Patch")

    def _taylor(self, el: Taylor, g: dict) -> None:
        """``TaylorP.{x_out…pz_out}`` hold ``term <coef> <e1>…<e6>`` monomials; the IR keeps a
        first-order 6×6 matrix, so only the linear and constant terms survive."""
        rows = ("x_out", "px_out", "y_out", "py_out", "z_out", "pz_out")
        matrix = [[0.0] * 6 for _ in range(6)]
        offset = [0.0] * 6
        dropped = 0
        for i, key in enumerate(rows):
            for term in _taylor_terms(g.get(key)):
                coef, exps = term
                order = sum(exps)
                if order == 0:
                    offset[i] = coef
                elif order == 1:
                    matrix[i][exps.index(1)] = coef
                else:
                    dropped += 1
        if any(any(r) for r in matrix) or any(offset):
            el.matrix, el.offset = matrix, offset
        if dropped:
            self.rep.lossy("PALS_TAYLOR_ORDER_TRUNCATED",
                           f"{dropped} Taylor term(s) above first order dropped (the IR keeps a "
                           "6×6 matrix)", element=el.name, kind="Taylor", terms=dropped)
        for key in ("S_q1_out", "S_qx_out", "S_qy_out", "S_qz_out"):
            if g.get(key):
                self.rep.lossy("PALS_SPIN_MAP_DROPPED", f"TaylorP.{key} (spin) has no IR field",
                               element=el.name, kind="Taylor")
                break

    def _begin_node(self, name: str, node: dict) -> bool:
        """A ``BeginningEle`` is not an IR element: it carries the branch's reference particle
        (``lattice-construction.md`` §Branch Expansion), so it is taken out of the line here and
        put back by the writer.  Returns True when *node* was one."""
        if str(node.get("kind")) != "BeginningEle":
            return False
        self.meta.setdefault("pals_beginning", name)
        for group in ("FloorP", "TwissP", "ParticleP", "TrackingP"):
            if node.get(group) is not None:
                self.meta.setdefault("pals_beginning_groups", {})[group] = node[group]
        g = node.get("ReferenceP") or {}
        self.rep.exact(name, "BeginningEle", code="PALS_BEGINNING_AS_REFERENCE",
                       message="BeginningEle read as the lattice reference particle")
        if self.reference is not None or not g:
            return True
        sp = ir_species(g.get("species_ref") or "") if g.get("species_ref") else None
        if g.get("species_ref") and sp is None:
            self.rep.lossy("PALS_SPECIES_UNKNOWN",
                           f"species_ref {g['species_ref']!r} is not a species lattix knows; "
                           "the reference falls back to a proton", element=name)
        sp = sp or species("proton")
        etot = self._num(g.get("E_tot_ref"), name, "E_tot_ref")
        pc = self._num(g.get("pc_ref"), name, "pc_ref")
        time = self._num(g.get("time_ref"), name, "time_ref") or 0.0
        if etot:
            self.reference = ReferenceParticle(species=sp, kinetic_energy_eV=etot - sp.mass_eV,
                                               time_s=time)
        elif pc:
            self.reference = ReferenceParticle.from_momentum(sp, pc, time_s=time)
        return True

    # -- reference and second pass -------------------------------------------
    @staticmethod
    def _placeholder_reference() -> ReferenceParticle:
        return ReferenceParticle(species=species("proton"), kinetic_energy_eV=1e9)

    def _reference(self, sp: str | Species | None, ke: float | None) -> ReferenceParticle:
        if self.reference is not None and sp is None and ke is None:
            return self.reference
        base = self.reference or self._placeholder_reference()
        if sp is None and ke is None:
            self.rep.equivalent("PALS_REFERENCE_ASSUMED",
                                "no BeginningEle/ReferenceP in the document; the reference "
                                "particle defaults to a 1 GeV proton (pass species=/"
                                "kinetic_energy_eV= to override)")
            return base
        return ReferenceParticle(
            species=species(sp) if sp is not None else base.species,
            kinetic_energy_eV=base.kinetic_energy_eV if ke is None else float(ke),
            rf_frequency_Hz=base.rf_frequency_Hz, time_s=base.time_s)

    def _second_pass(self) -> None:
        """Conversions that need the reference momentum at each element's entrance."""
        if not self.pending or self.lat.use is None:
            return
        todo = {id(el): job for el, job in self.pending}
        seen: dict[int, float] = {}
        for p in propagate(self.lat, warnings=self.warnings):
            job = todo.get(id(p.element))
            if job is None:
                continue
            ref = p.ref_in or self.lat.reference
            brho = ref.brho_signed
            first = seen.get(id(p.element))
            if first is not None:
                if abs(brho - first) > 1e-12 * max(1.0, abs(first)):
                    self.rep.equivalent(
                        "PALS_MULTI_ENERGY_NORMALIZED",
                        "this definition appears at two reference energies; its normalized "
                        "strengths were converted with the first occurrence's rigidity",
                        element=p.element.name, kind=p.element.kind, brho_first=first,
                        brho_here=brho)
                continue
            seen[id(p.element)] = brho
            self._apply_pending(p.element, job, ref, brho)

    def _apply_pending(self, el: Element, job: tuple[str, Any], ref: ReferenceParticle,
                       brho: float) -> None:
        what, data = job
        if what == "normalized":
            mp = el.multipole
            for kind, order, value, integrated in data:
                field = value * brho
                self._store_field(el, mp, "Bn" if kind == "Kn" else "Bs", order, field, integrated)
        elif what == "ksol":
            el.solenoid.Bsol_T = data * brho
        elif what == "kick":
            el.hkick, el.vkick = _kicks_from(data, brho, el.length)
        elif what == "bend_field":
            el.bend.angle = (data / brho) * el.length
        elif what == "refchange_total":
            el.energy_eV = data - ref.species.mass_eV


# --------------------------------------------------------------------------- module helpers

#: Groups kept verbatim in ``native["pals"]["groups"]`` so a pals→pals round trip is lossless.
_PASSTHROUGH_GROUPS = frozenset({
    "ElectricMultipoleP", "TwissP", "ParticleP", "ForkP", "ForkFromP", "GirderP", "ACKickerP",
    "BeamBeamP", "ConverterP", "FoilP", "CoordinateSetP", "FloorP",
})


def _merge_pals(parts: list[dict]) -> dict:
    """Combine ``PALS`` nodes subnode by subnode (``fundamentals.md`` §Load Files)."""
    out: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if key == "facility":
                pre, post = _split_expand(value if isinstance(value, list) else [value])
                cur_pre, cur_post = out.get("__pre", []), out.get("__post", [])
                out["__pre"], out["__post"] = cur_pre + pre, cur_post + post
            elif isinstance(value, list):
                out[key] = list(out.get(key, [])) + value
            elif isinstance(value, dict):
                merged = dict(out.get(key, {}))
                for k, v in value.items():
                    merged[k] = {**merged.get(k, {}), **v} if isinstance(v, dict) else v
                out[key] = merged
            else:
                out.setdefault(key, value)
    if "__pre" in out or "__post" in out:
        pre, post = out.pop("__pre", []), out.pop("__post", [])
        out["facility"] = pre + (["expand_lattice"] + post if post else [])
    return out


def _split_expand(facility: list) -> tuple[list, list]:
    for i, item in enumerate(facility):
        if item == "expand_lattice" or (isinstance(item, dict) and set(item) == {"expand_lattice"}):
            return facility[:i], facility[i + 1:]
    return facility, []


def _inherit(parent: dict, child: dict) -> dict:
    """``inherit:`` — the child's parameters override the parent's, group by group."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (parent or {}).items()}
    for key, value in (child or {}).items():
        if key in ("inherit", "repeat", "direction", "placement"):
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _modifiers(body: dict) -> tuple[dict, bool]:
    """``repeat``/``direction`` of a line item → ``(LineItem fields, reversed order?)``.

    ``beamlines.md``: a negative ``repeat`` reverses the element *order* only, while
    ``direction: -1`` is a true direction reversal (which is what ``LineItem.reverse`` means).
    """
    repeat = body.get("repeat", 1)
    try:
        repeat = int(repeat)
    except (TypeError, ValueError):
        repeat = 1
    direction = str(body.get("direction", 1)).strip()
    return {"repeat": abs(repeat) or 1, "reverse": direction in ("-1", "-1.0")}, repeat < 0


def _parse_multipole_key(key: str) -> tuple[str, int, bool] | None:
    """``Bn1`` → ``("Bn", 1, False)``; ``Kn3L`` → ``("Kn", 3, True)``; ``tilt7`` → ``("tilt", 7, False)``."""
    for prefix in ("tilt", "Bn", "Bs", "Kn", "Ks"):
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        integrated = rest.endswith("L") and prefix != "tilt"
        if integrated:
            rest = rest[:-1]
        if rest.isdigit() and (rest == "0" or not rest.startswith("0")):
            return prefix, int(rest), integrated
    return None


def _kicks_from(values: dict[str, float], brho: float, length: float) -> tuple[float, float]:
    """Order-0 multipole → IR deflections [rad] of the reference particle.

    ``parameters/magneticmultipole.md`` gives ``B_y + i·B_x = (Bn0 + i·Bs0)`` for ``N = 0``,
    and ``parameters/bend.md`` fixes the sign: a positive ``Kn0 = q·Bn0/p0`` bends towards
    **−x**.  The IR (like MAD-X) counts ``hkick`` positive towards +x, so
    ``hkick = −Kn0L`` and ``vkick = +Ks0L`` (a positive ``Bs0 = B_x`` pushes towards +y).
    """
    def total(normal: bool) -> float:
        out = 0.0
        for key, v in values.items():
            what, _order, integrated = _parse_multipole_key(key)  # type: ignore[misc]
            if what == "tilt" or (what in ("Bn", "Kn")) != normal:
                continue
            k = v if what in ("Kn", "Ks") else v / brho
            out += k if integrated else k * length
        return out

    return -total(True), total(False)


def _taylor_terms(node: Any) -> list[tuple[float, list[int]]]:
    """``term <coef> <e1> … <e6>`` lines of a ``TaylorP`` series."""
    if node is None:
        return []
    lines = node.splitlines() if isinstance(node, str) else (node if isinstance(node, list) else [])
    out: list[tuple[float, list[int]]] = []
    for line in lines:
        parts = str(line).split()
        if len(parts) != 8 or parts[0] != "term":
            continue
        try:
            out.append((float(parts[1]), [int(p) for p in parts[2:]]))
        except ValueError:
            continue
    return out


def read(path: str | Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
