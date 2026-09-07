"""Bmad writer (PLAN §6 task 2.2): table-driven, never silent.

Bmad is the **exact** target for an accelerating linac: its ``lcavity`` moves the
reference momentum, so — unlike the MAD-X writer — there is no ``energy_mode``
compromise.  Every normalized strength is written with the rigidity at *that*
element's entrance from :func:`lattix.ir.walk.propagate`, and Tao reproduces the
IR walk's reference energies exactly (tested in ``tests/formats/test_bmad_oracle.py``).

Two layouts:

``line_mode="nested"`` (default)
    the IR's :class:`~lattix.ir.lattice.Line` tree is emitted as
    ``name: line = (…)`` preserving sub-lines, ``3*sub`` repeats and ``-sub``
    reflections; ``use, <root>`` picks the root.
``line_mode="flat"``
    one ``line = (…)`` with every occurrence spelled out.

Conventions measured with Tao 20260828.0 (2026-09-03), never recalled:

* **a zero-length ``lcavity`` is a fatal error** for the default
  ``cavity_type = standing_wave`` (*"HAS ZERO LENGTH WHICH GIVES AN INFINITE
  PONDERMOTIVE KICK. SWITCH TO CAVITY_TYPE = TRAVELING_WAVE"*), and so is any
  ``0 < l < 1 mm`` for *either* cavity type (*"IS A LCAVITY WITH A SMALL (< 1 mm)
  BUT FINITE LENGTH"*).  A thin IR cavity is therefore written with ``l = 0``
  **and** ``cavity_type = traveling_wave``, which Bmad accepts and which
  reproduces ``ΔE = V·cos(2π·phi0)`` and the adiabatic damping exactly —
  ``EQUIVALENT:THIN_CAVITY_TRAVELING_WAVE``; a sub-millimetre cavity loses its
  length with ``LOSSY:SHORT_CAVITY_ZERO_LENGTH``.
* **a thin Bmad cavity has no transverse RF kick.**  Measured against
  HELIX/TraceWin on a 2.1 MeV proton linac: every entry of the 6×6 agrees to
  ~1e-12 *except* ``R21 = R43``, the thin gap's radial (pondermotive) RF
  defocusing, which a zero-length ``lcavity`` simply does not have (HELIX gives
  0.7617 / 0.6511 / 0.5649 for three 300 kV gaps at −30°).  The kick is written
  as a ``taylor`` lens right after the cavity (:func:`lattix.formats.base.with_rf_focusing`)
  and the cavity records ``EQUIVALENT:THIN_CAVITY_NO_RF_FOCUSING`` whenever
  ``V·sin(phase) ≠ 0``.
* **``phi0`` is in turns with ``ΔE = V·cos(2π·phi0)``** and is species
  independent — :func:`lattix.ir.rf.bmad_phi0` (measured: 1 MV at
  ``phi0 = −1/12`` gives ΔE = 866 025.40 eV).
* **rbend** — Bmad reads an rbend's ``l`` as the **chord** and shifts the pole
  faces by ``angle/2`` itself, so the writer emits
  ``l = L_arc·sin(θ/2)/(θ/2)`` and ``e1 = e1_ir − θ/2`` (measured round trip).
* **multipole** — ``k{n}l``/``k{n}sl`` are exactly MAD-X's ``knl``/``ksl``: a
  Bmad ``multipole, k1sl = 0.03`` and a MAD-X ``multipole, ksl = {0, 0.03}``
  give an identical map (cpymad vs Tao, transverse block equal to 0).  **Note
  that Bmad's own ``bmad_to_mad_sad_elegant -madx`` disagrees**: it writes
  ``ksl = −n!·a_n`` (``write_lattice_in_mad_format.f90``), and the deck it
  produces has a *reversed* skew quadrupole (measured: the two 4×4 blocks differ
  by 0.06 where the term is 0.03).  lattix follows the engines, not that
  converter; ``tests/formats/test_bmad_oracle.py`` pins the discrepancy so an
  upstream fix is noticed.
* **``k0l`` needs ``k0l_status``** — the default ``not_allowed`` makes a
  ``multipole`` with a dipole term fatal; the writer emits
  ``k0l_status = straight_reference`` (MAD-X's semantics: a kick that leaves the
  reference orbit and therefore the survey alone).
* **names** are limited to 40 characters (measured: 41 fails, and the manual
  says the same) and are case insensitive — see
  :mod:`lattix.formats.bmad.naming`.
* an **elliptical aperture needs all four limits**; Bmad raises
  *"FOR AN ELLIPTICAL APERTURE ALL FOUR X1_LIMIT … MUST BE SET"* otherwise, so
  the writer always emits the full quartet.
* a **species Bmad has no name for** becomes ``@M<mass in u><charge signs>``,
  whose atomic mass Bmad keeps to two decimals only (measured:
  ``@M0.938272088+`` is echoed as ``@M0.94+``) — ``LOSSY:SPECIES_MASS_ROUNDED``.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import note_quad_higher_orders
from lattix.formats.bmad.naming import NameMap, is_valid, name_tag
from lattix.ir.elements import (
    ALL_KINDS,
    ApertureP,
    Directive,
    Element,
    FieldMap,
    Freq,
    Superposition,
)
from lattix.ir.expr import ExpressionError, evaluate, identifiers
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import ReferenceParticle
from lattix.ir.rf import bmad_phi0
from lattix.ir.walk import propagate

#: lattix species name → Bmad's own spelling.  ``#1H-`` is the H-1 atom with one
#: extra electron; plain ``H-`` would use the isotope-averaged hydrogen mass.
SPECIES_TO_BMAD: dict[str, str] = {
    "proton": "proton", "antiproton": "antiproton", "electron": "electron",
    "positron": "positron", "deuteron": "deuteron", "h-": "#1H-",
}

#: IR ``Instrument.family`` → Bmad element key.
_FAMILY_TYPE = {"BPM": "monitor", "MONITOR": "monitor", "HMONITOR": "monitor",
                "VMONITOR": "monitor", "DETECTOR": "detector", "INSTRUMENT": "instrument",
                "PLACEHOLDER": "instrument"}

#: Twiss keys Bmad accepts under ``beginning[…]``.
_BEGINNING_KEYS = ("beta_a", "alpha_a", "beta_b", "alpha_b", "eta_x", "etap_x",
                   "eta_y", "etap_y")
#: lattix ``meta["twiss"]`` may use MAD-X spellings; map them onto Bmad's.
_TWISS_ALIAS = {"betx": "beta_a", "alfx": "alpha_a", "bety": "beta_b", "alfy": "alpha_b",
                "dx": "eta_x", "dpx": "etap_x", "dy": "eta_y", "dpy": "etap_y"}

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_EXPR_TOL = 1e-12
#: an element counts as accelerating above 1 µeV of reference gain
_ACCEL_TOL_eV = 1e-6
#: measured: Bmad refuses an ``lcavity`` with ``0 < l < 1 mm`` (lat_sanity_check)
_MIN_LCAVITY_L = 1e-3


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str                 # the Bmad construct this kind becomes
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


def _num(x: float) -> str:
    s = f"{float(x):.15g}"
    return "0" if s in ("-0", "-0.0") else s


#: Bmad built-in constants (``bmad/doc/expressions``): a deck variable with one of these names is
#: shadowed, so such variables are folded into numbers.
RESERVED_NAMES = frozenset({"pi", "twopi", "fourpi", "e_log", "sqrt_2", "degrees", "degrad", "raddeg",
                            "m_electron", "m_proton", "m_muon", "m_neutron", "m_deuteron", "m_pion_0",
                            "m_pion_charged", "c_light", "r_e", "r_p", "e_charge", "h_planck",
                            "h_bar_planck", "true", "false", "e_mass", "p_mass"})


def _reserved_user_names(lattice: Lattice) -> set[str]:
    return {n.lower() for n in lattice.variables if n.lower() in RESERVED_NAMES}


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`."""

    format = "bmad"

    RULES: dict[str, Rule] = {
        "Drift": Rule("drift"),
        "Quadrupole": Rule("quadrupole"),
        "Sextupole": Rule("sextupole"),
        "Octupole": Rule("octupole"),
        "Multipole": Rule("multipole"),
        "Bend": Rule("sbend/rbend"),
        "Solenoid": Rule("solenoid"),
        "RFCavity": Rule("lcavity"),
        "FieldMap": Rule("lcavity/solenoid/quadrupole/drift", "EQUIVALENT", "FM_TO_CAVITY",
                         "field map degraded per its integrated summary: RF → a thick lcavity "
                         "carrying the map's reference gain (FM_TO_CAVITY), static "
                         "solenoid/quadrupole → hard edge with drift padding "
                         "(FM_SOL_HARDEDGE / FM_QUAD_HARDEDGE), otherwise a drift"),
        "NCells": Rule("drift", "LOSSY", "NCELLS_TO_DRIFT",
                       "NCELLS cell train replaced by a drift of the same length"),
        "RFQCell": Rule("drift", "LOSSY", "RFQ_TO_DRIFT",
                        "RFQ cell replaced by a drift of the same length"),
        "Kicker": Rule("kicker"),
        "Collimator": Rule("rcollimator/ecollimator"),
        "Marker": Rule("marker"),
        "Instrument": Rule("monitor/instrument"),
        "Foil": Rule("foil"),
        "Taylor": Rule("taylor"),
        "Patch": Rule("patch"),
        "ReferenceChange": Rule("marker", "LOSSY", "REFCHANGE_DROPPED",
                                "Bmad's reference energy follows its own cavities; an explicit "
                                "reference change is written as a marker"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each Bmad cavity's rf_frequency attribute"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("superimpose children", "EQUIVALENT", "SUPERPOSITION_SUPERIMPOSED",
                              "overlapping fields written as superimposed Bmad elements"),
    }

    #: Directive roles Bmad can honour as a plain comment without losing anything.
    COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              use_expressions: bool = True, line_mode: str = "nested",
              line_name: str | None = None) -> FidelityReport:
        rep = FidelityReport(target_format="bmad", target_file=str(path))
        text = self.render(lattice, rep, use_expressions=use_expressions, line_mode=line_mode,
                           line_name=line_name)
        Path(path).write_text(text, encoding="utf-8")
        rep.raise_if(strict)
        return rep

    # ------------------------------------------------------------------
    def render(self, lattice: Lattice, rep: FidelityReport | None = None, *,
               use_expressions: bool = True, line_mode: str = "nested",
               line_name: str | None = None) -> str:
        if line_mode not in ("nested", "flat"):
            raise ValueError(f"line_mode must be 'nested' or 'flat', got {line_mode!r}")
        rep = rep if rep is not None else FidelityReport(target_format="bmad")
        placed = propagate(lattice)
        items = self._items(lattice, placed, rep)

        names = NameMap()
        root = lattice.use or lattice.name
        nested = line_mode == "nested" and root in lattice.lines
        line_names: dict[str, str] = {}
        root_name = names.reserve(line_name or root or "lat")
        if nested:
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
                                   "occurrence's rigidity is used for its normalized strengths",
                                   element=it.element.name, kind=it.element.kind,
                                   brho_first=brho0, brho_here=it.brho)
                continue
            defs[key] = (names.assign(it.element.name, it.element), it.element, it.brho)

        body: list[str] = [f"! lattix {__version__} from {lattice.meta.get('source_format', 'IR')}"]
        title = lattice.meta.get("bmad_title") or lattice.meta.get("madx_title")
        if title:
            body.append(f'title, "{title}"')
        body.append("")
        body.extend(self._header(lattice, rep))
        variables = self._variable_lines(lattice, use_expressions, rep)
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
            line = self._definition(name, el, brho, lattice, use_expressions, rep)
            if line:
                body.append(line)

        body.append("")
        body.append("! --- line")
        if nested:
            body.extend(self._nested_lines(root, root_name, line_names, lattice, defs, rep))
        else:
            body.extend(self._flat_line(root_name, items, defs, rep))
        body.append("")
        body.append(f"use, {root_name}")
        body.extend(self._superimpose_block(lattice, items, defs, rep))
        return "\n".join(body).rstrip("\n") + "\n"

    # -- flattening ---------------------------------------------------------
    def _items(self, lattice: Lattice, placed: list[Placed], rep: FidelityReport) -> list[_Item]:
        out: list[_Item] = []
        for p in placed:
            el = p.element
            ref: ReferenceParticle = p.ref_in or lattice.reference
            if isinstance(el, Superposition):
                self._record(rep, el, self.RULES["Superposition"])
                for offset, child_name in el.children:
                    child = lattice.elements.get(child_name)
                    if child is None:
                        rep.dropped("SUPERPOSITION_CHILD_MISSING",
                                    f"superposition child {child_name!r} is not defined",
                                    element=el.name, kind="Superposition")
                        continue
                    s0 = p.s_in + offset
                    if isinstance(child, FieldMap):     # the ladder applies inside a cluster too
                        out.extend(self._fieldmap_items(child, s0, ref, rep))
                        continue
                    out.append(_Item(child, s0, s0 + child.length, ref.brho_signed))
                continue
            if isinstance(el, FieldMap):
                out.extend(self._fieldmap_items(el, p.s_in, ref, rep))
                continue
            out.append(_Item(el, p.s_in, p.s_out, ref.brho_signed))
        return out

    @staticmethod
    def _fieldmap_items(el: FieldMap, s_in: float, ref: ReferenceParticle,
                        rep: FidelityReport) -> list[_Item]:
        """The field map's degradation ladder (PLAN §4.3).  Bmad keeps the RF map *thick*: an
        ``lcavity`` of the map's own length reproduces its reference gain and the adiabatic
        damping along it, where a zero-length one is a fatal Bmad error unless it is written
        ``cavity_type = traveling_wave`` (docs/oracles.md)."""
        r = replacement_for(el)
        rep.add(r.cls, r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, message in r.extra:
            rep.add(cls, code, message, element=el.name, kind="FieldMap")
        out, s = [], s_in
        for part in r.parts:
            out.append(_Item(part, s, s + part.length, ref.brho_signed))
            s += part.length
        return out

    # -- header -------------------------------------------------------------
    @staticmethod
    def _header(lattice: Lattice, rep: FidelityReport | None = None) -> list[str]:
        ref = lattice.reference
        sp = ref.species
        particle = SPECIES_TO_BMAD.get(sp.name.lower())
        out: list[str] = []
        if particle is None:
            # Bmad's bare-mass species is "@M<mass in u><charge signs>", but it keeps
            # only two decimals of the atomic mass (measured: "@M0.938272088+" is
            # stored and echoed as "@M0.94+"), so the mass is not exact.
            u = sp.mass_eV / 931_494_102.42
            signs = ("+" if sp.charge > 0 else "-") * abs(sp.charge) if sp.charge else ""
            particle = f"@M{u:.2f}{signs}"
            out.append(f"! lattix: species {sp.name!r} mass = {_num(sp.mass_eV)} eV, "
                       f"charge = {sp.charge}")
            if rep is not None:
                rep.lossy("SPECIES_MASS_ROUNDED",
                          f"Bmad has no name for species {sp.name!r}; written as {particle!r}, "
                          "whose atomic mass Bmad rounds to two decimals",
                          element=None, kind=None, mass_eV=sp.mass_eV, charge=sp.charge,
                          written=particle)
        out.append(f"parameter[particle] = {particle}")
        out.append(f"parameter[e_tot] = {_num(ref.total_energy_eV)}")
        geometry = str(lattice.meta.get("bmad_parameter", {}).get("geometry", "open")).strip()
        out.append(f"parameter[geometry] = {geometry or 'open'}")
        twiss = lattice.meta.get("twiss") or {}
        emitted = {}
        for key, value in twiss.items():
            k = _TWISS_ALIAS.get(str(key).lower(), str(key).lower())
            if k in _BEGINNING_KEYS:
                emitted[k] = float(value)
        for k in _BEGINNING_KEYS:
            if k in emitted:
                out.append(f"beginning[{k}] = {_num(emitted[k])}")
        return out

    def _variable_lines(self, lattice: Lattice, use_expressions: bool,
                        rep: FidelityReport) -> list[str]:
        if not use_expressions or not lattice.variables:
            return []
        keep = {}
        for name, var in lattice.variables.items():
            if name.lower() in RESERVED_NAMES:
                rep.equivalent("VARIABLE_DROPPED",
                               f"variable {name!r} collides with a Bmad built-in constant; its value "
                               "is folded into the numbers", element=None, kind=None, variable=name)
            elif is_valid(name):
                keep[name] = var
            else:
                rep.equivalent("VARIABLE_DROPPED",
                               f"variable {name!r} is not a writable Bmad identifier; its value "
                               "is folded into the numbers",
                               element=None, kind=None, variable=name)
        reserved = _reserved_user_names(lattice)
        lines = []
        for name in self._topo(keep):
            var = keep[name]
            expr = var.expression
            if expr is not None and expr.text.strip() and self._expr_ok(expr.text, keep) \
                    and not (identifiers(expr.text) & reserved):
                lines.append(f"{name} = {expr.text.strip()}")
            else:
                lines.append(f"{name} = {_num(var.value)}")
        return lines

    @staticmethod
    def _expr_ok(text: str, keep: dict) -> bool:
        """True when *text* only names variables Bmad will also have."""
        try:
            evaluate(text, {k: v.value for k, v in keep.items()})
        except ExpressionError:
            return False
        return True

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
    def _definition(self, name: str, el: Element, brho: float, lattice: Lattice,
                    use_expressions: bool, rep: FidelityReport) -> str | None:
        rule = self.RULES.get(el.kind)
        if rule is None:                                    # pragma: no cover - RULES is total
            raise KeyError(f"Bmad writer has no rule for kind {el.kind!r}")
        if isinstance(el, Freq):
            rep.exact(el.name, "Freq", message=rule.message)
            return None
        if isinstance(el, Directive):
            if el.role in self.COMMENT_ROLES:
                rep.exact(el.name, "Directive", code="DIRECTIVE_AS_COMMENT",
                          message=f"role {el.role!r} carries no Bmad physics; written as a comment")
            else:
                rep.dropped(rule.code, rule.message, element=el.name, kind="Directive",
                            card=el.card, role=el.role)
            args = " ".join(str(a) for a in el.args)
            return f"! lattix directive ({el.format} {el.role}): {el.card} {args}".rstrip()
        base, attrs = getattr(self, f"_def_{el.kind.lower()}")(el, brho, lattice,
                                                               use_expressions, rep)
        attrs = list(attrs) + self._aperture_attrs(el) + self._shift_attrs(el, base)
        self._record(rep, el, rule)
        original = (el.provenance.original_name if el.provenance else None) or el.name
        tag = ("   " + name_tag(original, el.provenance.original_type if el.provenance else None)
               if name != original else "")
        joined = ", ".join(attrs)
        return f"{name}: {base}" + (f", {joined}" if joined else "") + tag

    @staticmethod
    def _record(rep: FidelityReport, el: Element, rule: Rule) -> None:
        if rule.cls == "EXACT":
            rep.exact(el.name, el.kind)
        else:
            rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind,
                    target=rule.target)

    @staticmethod
    def _aperture_attrs(el: Element) -> list[str]:
        """Bmad needs all four limits for an elliptical aperture (measured)."""
        ap: ApertureP | None = el.aperture
        if ap is None or (ap.x_limits is None and ap.y_limits is None):
            return []
        if el.kind == "Collimator":
            return []                       # written by _def_collimator itself
        x = ap.x_limits or ap.y_limits
        y = ap.y_limits or ap.x_limits
        at = {"ENTRANCE": "entrance_end", "EXIT": "exit_end", "BOTH_ENDS": "both_ends",
              "CONTINUOUS": "continuous"}[ap.aperture_at]
        kind = "elliptical" if ap.shape == "ELLIPTICAL" else "rectangular"
        return [f"x1_limit = {_num(-x[0])}", f"x2_limit = {_num(x[1])}",
                f"y1_limit = {_num(-y[0])}", f"y2_limit = {_num(y[1])}",
                f"aperture_type = {kind}", f"aperture_at = {at}"]

    @staticmethod
    def _shift_attrs(el: Element, base: str) -> list[str]:
        s = el.shift
        if s is None or s.is_zero() or base == "patch":
            return []
        out = []
        # IR y_rot, a rotation about y, is Bmad's x_pitch; IR x_rot is -y_pitch (the reader's mirror)
        for attr, value in (("x_offset", s.x_offset), ("y_offset", s.y_offset),
                            ("z_offset", s.z_offset), ("x_pitch", s.y_rot),
                            ("y_pitch", -s.x_rot)):
            if value:
                out.append(f"{attr} = {_num(value)}")
        if s.tilt:
            out.append(f"{'roll' if base in ('sbend', 'rbend') else 'tilt'} = {_num(s.tilt)}")
        return out

    def _attr(self, el: Element, bmad_attr: str, ir_path: str, value: float,
              lattice: Lattice, use_expressions: bool, rep: FidelityReport) -> str:
        """``k1 = 0.6``, or ``k1 = kqf`` when the stored expression still gives *value*.

        Bmad variables are always live (there is no ``:=``), so a deferred MAD-X
        expression becomes a plain Bmad assignment.  The stored text is the source
        format's *normalized* quantity, which for k1/ks/knl/angle/l is exactly the
        Bmad quantity too, so the check is done on the number this call would print.
        """
        if use_expressions:
            text = (el.native.get("bmad", {}).get(f"{bmad_attr}_expr")
                    or el.native.get("madx", {}).get(f"{bmad_attr}_expr"))
            expr = el.expressions.get(ir_path)
            if text is None and expr is not None:
                text = expr.text
            if text and not isinstance(text, (list, tuple)) and identifiers(text) & _reserved_user_names(lattice):
                text = None          # a user variable named like a Bmad constant: write the number
            if text and not isinstance(text, (list, tuple)):
                variables = {k: v.value for k, v in lattice.variables.items()}
                try:
                    got = evaluate(str(text), variables)
                except ExpressionError:
                    got = None
                if got is not None and abs(got - value) <= _EXPR_TOL * max(1.0, abs(value)):
                    return f"{bmad_attr} = {str(text).strip()}"
                why = ("cannot be evaluated by lattix (element-attribute or macro reference)"
                       if got is None else "no longer matches the IR value")
                rep.equivalent("EXPRESSION_DROPPED",
                               f"{bmad_attr} expression {str(text)!r} {why}; "
                               "the number is written instead",
                               element=el.name, kind=el.kind, attr=bmad_attr, value=value)
        return f"{bmad_attr} = {_num(value)}"

    # per-kind builders: (bmad element key, [attribute strings])
    def _def_drift(self, el, brho, lat, ux, rep):
        return "drift", [self._attr(el, "l", "length", el.length, lat, ux, rep)]

    def _def_quadrupole(self, el, brho, lat, ux, rep):
        note_quad_higher_orders(el, rep, "Bmad")
        attrs = [self._attr(el, "l", "length", el.length, lat, ux, rep),
                 self._attr(el, "k1", "multipole.Bn[1]",
                            el.multipole.Bn.get(1, 0.0) / brho, lat, ux, rep)]
        if el.multipole.tilt.get(1):
            attrs.append(self._attr(el, "tilt", "multipole.tilt[1]",
                                    el.multipole.tilt[1], lat, ux, rep))
        return "quadrupole", attrs

    def _def_sextupole(self, el, brho, lat, ux, rep):
        return "sextupole", self._thick_multipole(el, brho, lat, ux, rep, 2, "k2")

    def _def_octupole(self, el, brho, lat, ux, rep):
        return "octupole", self._thick_multipole(el, brho, lat, ux, rep, 3, "k3")

    def _thick_multipole(self, el, brho, lat, ux, rep, order, kn):
        attrs = [self._attr(el, "l", "length", el.length, lat, ux, rep),
                 self._attr(el, kn, f"multipole.Bn[{order}]",
                            el.multipole.Bn.get(order, 0.0) / brho, lat, ux, rep)]
        if el.multipole.Bs.get(order):
            # a skew thick magnet is the normal one rotated by -atan2(Bs, Bn)/(order+1)
            bn, bs = el.multipole.Bn.get(order, 0.0), el.multipole.Bs[order]
            attrs[1] = f"{kn} = {_num(math.hypot(bn, bs) / brho)}"
            tilt = el.multipole.tilt.get(order, 0.0) - math.atan2(bs, bn) / (order + 1)
            attrs.append(f"tilt = {_num(tilt)}")
            rep.equivalent("SKEW_AS_TILT",
                           f"skew component B{order} folded into one rotated normal magnet (exact)",
                           element=el.name, kind=el.kind, order=order)
        elif el.multipole.tilt.get(order):
            attrs.append(self._attr(el, "tilt", f"multipole.tilt[{order}]",
                                    el.multipole.tilt[order], lat, ux, rep))
        return attrs

    def _def_multipole(self, el, brho, lat, ux, rep):
        attrs = []
        if el.length:
            attrs.append(f"l = {_num(el.length)}")
            rep.equivalent("THIN_MULTIPOLE_LENGTH",
                           "Bmad's 'multipole' is a thin element; the IR length is dropped and the "
                           "integrated strengths are kept",
                           element=el.name, kind="Multipole", length=el.length)
            attrs.pop()
        for order in sorted(set(el.multipole.BnL) | set(el.multipole.BsL)):
            bn = el.multipole.BnL.get(order, 0.0)
            bs = el.multipole.BsL.get(order, 0.0)
            if bn:
                attrs.append(f"k{order}l = {_num(bn / brho)}")
            if bs:
                attrs.append(f"k{order}sl = {_num(bs / brho)}")
        if el.multipole.BnL.get(0) or el.multipole.BsL.get(0):
            # measured: a k0l with the default k0l_status = not_allowed is a FATAL error;
            # 'straight_reference' is MAD-X's semantics (a kick that does not bend the
            # reference orbit, so the survey is unchanged)
            attrs.append("k0l_status = straight_reference")
        if el.multipole.tilt.get(0):
            attrs.append(f"tilt = {_num(el.multipole.tilt[0])}")
        return "multipole", attrs

    def _def_bend(self, el, brho, lat, ux, rep):
        b = el.bend
        base = "rbend" if b.rect else "sbend"
        half = b.angle / 2.0 if b.rect else 0.0
        length = el.length
        if b.rect and b.angle:
            length = el.length * math.sin(b.angle / 2.0) / (b.angle / 2.0)     # arc -> chord
        attrs = [f"l = {_num(length)}" if b.rect
                 else self._attr(el, "l", "length", length, lat, ux, rep),
                 self._attr(el, "angle", "bend.angle", b.angle, lat, ux, rep)]
        if (b.e1 - half) or (b.e2 - half):
            attrs.append(self._attr(el, "e1", "bend.e1", b.e1 - half, lat, ux, rep))
            attrs.append(self._attr(el, "e2", "bend.e2", b.e2 - half, lat, ux, rep))
        if b.edge_int1:
            attrs.append(f"fint = {_num(b.edge_int1)}")
        if b.edge_int2 is not None:
            attrs.append(f"fintx = {_num(b.edge_int2)}")
        if b.hgap:
            attrs.append(f"hgap = {_num(b.hgap)}")
        if el.multipole.Bn.get(1):
            attrs.append(self._attr(el, "k1", "multipole.Bn[1]",
                                    el.multipole.Bn[1] / brho, lat, ux, rep))
        if el.multipole.Bn.get(2):
            attrs.append(self._attr(el, "k2", "multipole.Bn[2]",
                                    el.multipole.Bn[2] / brho, lat, ux, rep))
        if b.tilt_ref:
            attrs.append(f"ref_tilt = {_num(b.tilt_ref)}")
        if b.fringe_k2 is not None:
            rep.equivalent("FRINGE_K2_DROPPED",
                           "the TraceWin second fringe coefficient K2 has no Bmad attribute",
                           element=el.name, kind="Bend", fringe_k2=b.fringe_k2)
        return base, attrs

    def _def_solenoid(self, el, brho, lat, ux, rep):
        return "solenoid", [self._attr(el, "l", "length", el.length, lat, ux, rep),
                            self._attr(el, "ks", "solenoid.Bsol_T",
                                       el.solenoid.Bsol_T / brho, lat, ux, rep)]

    def _def_rfcavity(self, el, brho, lat, ux, rep):
        return self._cavity(el, el.rf, el.length, lat, ux, rep)

    def _cavity(self, el, rf, length, lat, ux, rep, *, voltage: float | None = None):
        volt = voltage if voltage is not None else rf.voltage_V
        if voltage is None and not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else length)
        attrs = [f"l = {_num(length)}",
                 self._attr(el, "voltage", "rf.voltage_V", volt, lat, ux, rep),
                 f"phi0 = {_num(bmad_phi0(rf.phase_rad))}"]
        if rf.frequency_Hz:
            attrs.append(self._attr(el, "rf_frequency", "rf.frequency_Hz",
                                    rf.frequency_Hz, lat, ux, rep))
        if rf.L_active_m is not None and rf.L_active_m != length and length:
            rep.equivalent("L_ACTIVE_DEPENDENT",
                           "Bmad computes l_active from l and n_cell; the IR's own active length "
                           "is recorded but not written",
                           element=el.name, kind=el.kind, L_active_m=rf.L_active_m)
        if 0.0 < length < _MIN_LCAVITY_L:
            # measured: Bmad rejects 0 < L < 1 mm outright ("IS A LCAVITY WITH A SMALL
            # (< 1 mm) BUT FINITE LENGTH"), for either cavity_type.
            rep.lossy("SHORT_CAVITY_ZERO_LENGTH",
                      f"Bmad rejects an lcavity with 0 < l < {_MIN_LCAVITY_L} m; the "
                      f"{length} m length is dropped and the cavity written as a thin "
                      "traveling-wave gap (the lattice is shorter by that much)",
                      element=el.name, kind=el.kind, length=length)
            attrs[0] = "l = 0"
            length = 0.0
        if length == 0.0:
            # measured: a zero-length standing-wave lcavity is a FATAL Bmad error
            attrs.append("cavity_type = traveling_wave")
            rep.equivalent("THIN_CAVITY_TRAVELING_WAVE",
                           "Bmad rejects a zero-length standing-wave lcavity (infinite pondermotive "
                           "kick); written with l = 0 and cavity_type = traveling_wave, which "
                           "reproduces dE = V*cos(2*pi*phi0) and the adiabatic damping exactly",
                           element=el.name, kind=el.kind)
            if volt and abs(math.sin(rf.phase_rad)) > 1e-12:
                # measured against the HELIX/TraceWin thin-gap model on a 2.1 MeV proton
                # linac: every entry of the 6x6 agrees to ~1e-15 EXCEPT R21 = R43, the thin
                # gap's radial RF defocusing, which a zero-length Bmad lcavity does not have.
                rep.equivalent("THIN_CAVITY_NO_RF_FOCUSING",
                               "a zero-length Bmad lcavity applies no transverse RF (pondermotive) "
                               "kick of its own; the TraceWin thin-gap defocusing R21 = R43 is carried "
                               "by the 'taylor' lens written after the cavity",
                               element=el.name, kind=el.kind, voltage_V=volt,
                               phase_rad=rf.phase_rad)
        else:
            if rf.cavity_type == "TRAVELING_WAVE":
                attrs.append("cavity_type = traveling_wave")
            if rf.n_cell is not None:
                attrs.append(f"n_cell = {int(rf.n_cell)}")
            elif rf.frequency_Hz and length < 0.5 * 299_792_458.0 / rf.frequency_Hz:
                # Bmad warns that l_active collapses to zero; n_cell = 0 is its own remedy
                attrs.append("n_cell = 0")
        return "lcavity", attrs

    def _def_fieldmap(self, el, brho, lat, ux, rep):   # pragma: no cover - expanded in _items
        return "drift", [f"l = {_num(el.length)}"]

    def _def_ncells(self, el, brho, lat, ux, rep):
        return "drift", [f"l = {_num(el.length)}"]

    _def_rfqcell = _def_ncells

    def _def_kicker(self, el, brho, lat, ux, rep):
        if el.electric:
            rep.lossy("EKICK_AS_MAGNETIC",
                      "electric steerer written as a magnetic Bmad kicker",
                      element=el.name, kind="Kicker")
        return "kicker", [f"l = {_num(el.length)}",
                          self._attr(el, "hkick", "hkick", el.hkick, lat, ux, rep),
                          self._attr(el, "vkick", "vkick", el.vkick, lat, ux, rep)]

    def _def_collimator(self, el, brho, lat, ux, rep):
        ap = el.aperture
        base = "ecollimator" if (ap is not None and ap.shape == "ELLIPTICAL") else "rcollimator"
        attrs = [f"l = {_num(el.length)}"]
        if ap is not None and (ap.x_limits or ap.y_limits):
            x = ap.x_limits or ap.y_limits
            y = ap.y_limits or ap.x_limits
            attrs += [f"x1_limit = {_num(-x[0])}", f"x2_limit = {_num(x[1])}",
                      f"y1_limit = {_num(-y[0])}", f"y2_limit = {_num(y[1])}"]
        return base, attrs

    def _def_marker(self, el, brho, lat, ux, rep):
        return "marker", ([] if not el.length
                          else [self._attr(el, "l", "length", el.length, lat, ux, rep)])

    def _def_instrument(self, el, brho, lat, ux, rep):
        return (_FAMILY_TYPE.get(el.family.upper(), "instrument"),
                [self._attr(el, "l", "length", el.length, lat, ux, rep)])

    def _def_foil(self, el, brho, lat, ux, rep):
        attrs = [f'material_type = "{el.material}"']
        if el.thickness_kg_per_m2:
            attrs.append(f"area_density = {_num(el.thickness_kg_per_m2)}")
        else:
            nat = el.native.get("bmad", {})
            if "thickness" in nat:
                attrs.append(f"thickness = {_num(float(nat['thickness']))}")
            else:
                rep.lossy("FOIL_THICKNESS_UNKNOWN",
                          "the IR foil has neither an areal density nor a Bmad thickness",
                          element=el.name, kind="Foil")
        return "foil", attrs

    def _def_taylor(self, el, brho, lat, ux, rep):
        attrs = [f"l = {_num(el.length)}"]
        for i in range(6):
            if el.offset[i]:
                attrs.append(f"{{{i + 1}: {_num(el.offset[i])}|}}")
            for j in range(6):
                v = el.matrix[i][j]
                if v != (1.0 if i == j else 0.0):
                    attrs.append(f"{{{i + 1}: {_num(v)}|{j + 1}}}")
        return "taylor", attrs

    def _def_patch(self, el, brho, lat, ux, rep):
        attrs = []
        for attr, value in (("x_offset", el.x_offset), ("y_offset", el.y_offset),
                            ("z_offset", el.z_offset), ("x_pitch", el.y_rot),
                            ("y_pitch", -el.x_rot), ("tilt", el.tilt)):
            if value:
                attrs.append(f"{attr} = {_num(value)}")
        return "patch", attrs

    def _def_referencechange(self, el, brho, lat, ux, rep):
        return "marker", []

    def _def_superposition(self, el, brho, lat, ux, rep):   # pragma: no cover - expanded in _items
        return "marker", []

    # -- lines ---------------------------------------------------------------
    def _flat_line(self, root_name: str, items: list[_Item], defs: dict,
                   rep: FidelityReport) -> list[str]:
        refs = [defs[id(it.element)][0] for it in items
                if not isinstance(it.element, (Freq, Directive))]
        if not refs:
            rep.dropped("EMPTY_LINE", "the lattice has no element Bmad can place")
            return [f"{root_name}_nil: marker", f"{root_name}: line = ({root_name}_nil)"]
        return [self._wrap(f"{root_name}: line = (", refs, ")")]

    def _nested_lines(self, root: str, root_name: str, line_names: dict[str, str],
                      lattice: Lattice, defs: dict, rep: FidelityReport) -> list[str]:
        by_name = {el.name: nm for nm, el, _ in defs.values()}
        sup_lines: list[str] = []
        nil: list[str] = []

        def emits_nothing(ref: str) -> bool:
            return isinstance(lattice.elements.get(ref), (Freq, Directive))

        def nil_marker() -> str:
            if not nil:
                nil.append(f"{root_name}_nil")
            return nil[0]

        def ref_name(ref: str) -> str:
            if ref in line_names:
                return line_names[ref]
            el = lattice.elements.get(ref)
            if isinstance(el, FieldMap):          # a padded hard edge is a sub-line of its parts
                parts = replacement_for(el).parts
                if len(parts) > 1:
                    nm = re.sub(r"[^A-Za-z0-9_]", "_", f"{ref}_fm").lower()
                    sup_lines.append(f"{nm}: line = ({', '.join(by_name[q.name] for q in parts if q.name in by_name)})")
                    return nm
                return by_name.get(parts[0].name, ref)
            if ref in by_name:
                return by_name[ref]
            if isinstance(el, Superposition):     # children were emitted, the parent was not
                nm = re.sub(r"[^A-Za-z0-9_]", "_", ref).lower()
                kids = [by_name[c] for _, c in el.children if c in by_name]
                if not kids:                      # Bmad rejects an empty line
                    rep.dropped("EMPTY_LINE",
                                f"superposition {ref!r} has no writable child; a marker keeps "
                                "the line non-empty",
                                element=ref, kind="Superposition")
                    kids = [nil_marker()]
                sup_lines.append(f"{nm}: line = ({', '.join(kids)})")
                return nm
            return ref

        def item_text(it) -> str:
            txt = ref_name(it.ref)
            if it.repeat != 1:
                txt = f"{it.repeat}*{txt}"
            if it.reverse:
                txt = f"-{txt}"
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
            body = [item_text(it) for it in kept] or [nil_marker()]
            emitted.append(self._wrap(f"{line_names.get(name, name)}: line = (", body, ")"))

        emit(root)
        head = [f"{nil[0]}: marker"] if nil else []
        return head + sup_lines + emitted

    @staticmethod
    def _wrap(prefix: str, items: list[str], suffix: str, width: int = 110) -> str:
        """Bmad continues a line that ends in a comma, so wrap on commas."""
        out, cur = [], prefix
        pad = " " * min(len(prefix), 20)
        for i, item in enumerate(items):
            piece = item + ("," if i < len(items) - 1 else "")
            if len(cur) + len(piece) + 1 > width and cur.strip() not in (prefix.strip(),):
                out.append(cur.rstrip())
                cur = pad
            cur += piece + (" " if i < len(items) - 1 else "")
        return "\n".join([*out, cur.rstrip() + suffix])

    # -- superimpose ----------------------------------------------------------
    @staticmethod
    def _superimpose_block(lattice: Lattice, items: list[_Item], defs: dict,
                           rep: FidelityReport) -> list[str]:
        """Re-emit a reader-preserved ``native['bmad']['superimpose']`` verbatim."""
        out: list[str] = []
        seen: set[int] = set()
        for it in items:
            el = it.element
            if id(el) in seen:
                continue
            seen.add(id(el))
            sup = el.native.get("bmad", {}).get("superimpose")
            if not sup:
                continue
            name = defs[id(el)][0]
            parts = ", ".join(f"{k} = {v}" for k, v in sup.items() if v not in (None, ""))
            # never re-emit `name[superimpose] = T`: the element is already in the line,
            # and Bmad would then place a second copy of it.
            out.append(f"! lattix: {name} was superimposed in the source deck"
                       + (f" ({parts})" if parts else "")
                       + "; it is written in the line at the position the reader resolved")
        return [""] + out if out else []


_MISSING = set(ALL_KINDS) - set(Writer.RULES)
if _MISSING:                                        # pragma: no cover - guarded by a test too
    raise RuntimeError(f"Bmad writer RULES do not cover {sorted(_MISSING)}")


def render(lattice: Lattice, **options) -> str:
    return Writer().render(lattice, **options)


def write(lattice: Lattice, path: str | Path, **options) -> FidelityReport:
    return Writer().write(lattice, Path(path), **options)
