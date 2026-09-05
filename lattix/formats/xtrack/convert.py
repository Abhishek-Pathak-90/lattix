"""IR ⇄ ``xtrack.Line`` conversion (PLAN §6 task 3.5), table driven and never silent.

Everything below was **measured** on this machine against xtrack 0.103.5 (base env)
and 0.112.0 (env ``lattix``); the two releases expose the same ``_xofields`` for every
class used here, so one code path serves both.

Conventions pinned by measurement (2026-09-03)
----------------------------------------------
* **Cavity phase.**  ``Cavity`` carries *two* phase attributes whose effects add:
  ``lag`` in degrees and ``phase`` in radians (``xtrack/mad_writer.py`` writes MAD-X
  ``lag = lag/360 + phase/2π``).  The IR's synchronous phase φ (0 = crest, cos
  convention) maps to ``lag_deg = φ·180/π + 90`` ≡ ``phase = φ + π/2``.  Measured:
  ``Cavity(voltage=1 MV, lag=60°)`` on a 2.1 MeV proton gives ΔE = 866 025.4038 eV =
  ``+V·cos(30°)`` to 5e-14 relative in **both** 0.103.5 and 0.112.0.  The writer emits
  ``phase`` (radians) because ``lag`` is deprecated in 0.112 and ``phase`` survives
  ``to_dict``/``from_dict`` unchanged in 0.103.5; the reader accepts both and adds them.
* **Kicker.**  ``Multipole(knl=[-hkick], ksl=[+vkick])`` deflects by ``px += +hkick``,
  ``py += +vkick`` (measured: knl=[-1e-3] → px = +1.000000e-03).  This is the same
  convention ``xtrack/mad_writer.py`` uses in reverse (``hkick = -knl[0]``).
* **Tilt.**  ``rot_s_rad`` is exactly MAD-X ``tilt``: a quadrupole ``k1=0.6, l=0.3``
  with ``rot_s_rad=0.3`` reproduces cpymad's ``tilt=0.3`` sector map to 2.2e-16, and
  equals ``srot(−θ)·M·srot(+θ)`` to machine precision.  ``(k1, k1s)`` is *not* the same
  thing in xtrack (its combined thick model differs by 7.6e-4 on that element), so the
  IR's ``multipole.tilt[n]`` becomes ``rot_s_rad`` and only an explicit skew component
  ``multipole.Bs[n]`` becomes ``k1s``/``k2s``/``k3s``.
* **Misalignment.**  ``shift_x``/``shift_y``/``shift_s`` are MAD-X ``dx``/``dy``/``ds``
  and ``rot_s_rad_no_frame`` is ``dpsi`` (measured identical to ``rot_s_rad`` for a
  quadrupole).  ``Drift`` and ``Marker`` have no such fields, but neither is affected
  by a shift or a roll, so a body shift on them is recorded ``EQUIVALENT:SHIFT_NO_OP``
  instead of wrapping them in an ``XYShift``/``SRotation`` sandwich.
* **Frame elements.**  ``XYShift(dx)`` maps ``x → x − dx``; ``SRotation(angle_deg)``
  rotates the *frame* by +angle; ``ZetaShift(dzeta)`` maps ``ζ → ζ − dzeta``.  All
  three match the IR ``Patch`` sign convention used by :func:`lattix.ir.lattice.survey`.
* **Bend.**  ``h`` cannot be assigned directly in either release ("Setting `h` directly
  is not allowed"): the writer passes ``length`` (arc) and ``angle``, which sets
  ``h = angle/length`` and ``k0_from_h = True``.  Total pole-face angles go to
  ``edge_entry_angle``/``edge_exit_angle``, so a rectangular IR bend is emitted as a
  sector ``xt.Bend`` with e1/e2 already containing angle/2 — exact, and free of the
  ``RBend`` chord/sagitta model.

Reference energy
----------------
xtrack keeps ``p0c`` constant through RF (energy goes into ``delta``), exactly like
MAD-X, so accelerating lattices get an ``EQUIVALENT:CONST_P0`` entry and the same
``energy_mode`` switch the MAD-X writer has:

``energy_mode="delta"`` (default)
    normalized strengths use the rigidity xtrack's own reference particle has there —
    the start rigidity across the RF gains the line contains (xtrack puts them into
    ``delta``), the local rigidity across reference changes it cannot apply; kicks,
    ``FirstOrderTaylorMap``s and a bend's ``k0`` are rescaled by the same ratio
    (:mod:`lattix.ir.energy_mode`; ``EQUIVALENT:CONST_P0_DELTA_RIGIDITY``).
``energy_mode="local"``
    normalized strengths use Bρ at *that element's* entrance
    (``EQUIVALENT:CONST_P0_LOCAL_RIGIDITY``).
``energy_mode="constant"``
    one Bρ, the lattice start (``EQUIVALENT:CONST_P0_START_RIGIDITY``).

Provenance
----------
``line.metadata["lattix"]`` records the writer version, the source format, the energy
mode and one row per emitted xtrack element (IR name, IR kind, role) — the ``! lattix``
comment tag other writers put in the text deck.  :func:`from_line` uses it to restore
the exact IR kinds and names when it is present and falls back to xtrack's own
conventions when it is not (a foreign line from ``xt.load`` or ``from_madx_sequence``).
"""
from __future__ import annotations

import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any

from lattix._version import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import note_quad_higher_orders
from lattix.ir.elements import (
    ALL_KINDS,
    RFP,
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
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    NCells,
    Octupole,
    Patch,
    Provenance,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Sextupole,
    Solenoid,
    SolenoidP,
    Superposition,
    Taylor,
)
from lattix.ir.energy_mode import (
    check_mode,
    mode_ratio,
    phase_slip_turns,
    probe_momentum_ratio,
    record_phase_slip,
    record_rigidity_mode,
    rigidity_for,
    scale_taylor,
    slip_is_zero,
    undo_phase_slip,
)
from lattix.ir.lattice import Lattice, Line, LineItem, Placed
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.units import C_LIGHT
from lattix.ir.walk import energy_gain_eV, propagate


#: an element counts as accelerating above 1 µeV of reference gain (same tolerance as
#: the MAD-X writer: far below any physical gain, far above ``cos(±π/2)`` noise).
def allow_jit() -> None:
    """Let xtrack compile its kernels just in time.

    xtrack >= 0.112 refuses to build a tracker unless xsuite's prebuilt kernels are installed or
    just-in-time compilation is explicitly allowed; lattix only needs a C compiler, so it opts in
    (environment variable for a fresh import, ``xobjects.settings`` when xobjects is already loaded).
    """
    os.environ.setdefault("XSUITE_ALLOW_KERNEL_COMPILATION", "1")
    try:
        import xobjects as xo
    except ImportError:  # pragma: no cover - xtrack absent
        return
    settings = getattr(xo, "settings", None)
    if settings is not None and hasattr(settings, "allow_kernel_compilation"):
        settings.allow_kernel_compilation = True


_ACCEL_TOL_eV = 1e-6

#: characters xtrack tolerates in an element name and MAD-X can still read back.
_BAD_NAME = re.compile(r"[^A-Za-z0-9_.$]")
_LEADING = re.compile(r"^[^A-Za-z_]+")

#: metadata key holding lattix provenance inside ``line.metadata``.
METADATA_KEY = "lattix"

#: Directive roles that carry no physics, so a marker loses nothing.
COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
def sanitize(name: str) -> str:
    """An xtrack-safe element name (xtrack keys its element dict by name, so the only
    hard requirement is uniqueness; the character set keeps MAD-X export working)."""
    out = _BAD_NAME.sub("_", str(name).strip())
    out = _LEADING.sub("", out)
    return out or "el"


class NameMap:
    """Sanitize + uniquify: ``q1``, ``q1_2``, ``q1_3`` … (xtrack requires unique names)."""

    def __init__(self) -> None:
        self.used: set[str] = set()
        self.renamed: dict[str, str] = {}

    def assign(self, name: str) -> str:
        base = sanitize(name)
        nm = base
        k = 2
        while nm in self.used:
            nm = f"{base}_{k}"
            k += 1
        self.used.add(nm)
        if nm != name:
            self.renamed[nm] = name
        return nm


# ---------------------------------------------------------------------------
# capability matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str                 # the xtrack construct this kind becomes
    cls: str = "EXACT"          # default fidelity class
    code: str = "OK"
    message: str = ""


RULES: dict[str, Rule] = {
    "Drift": Rule("xt.Drift"),
    "Quadrupole": Rule("xt.Quadrupole"),
    "Sextupole": Rule("xt.Sextupole"),
    "Octupole": Rule("xt.Octupole"),
    "Multipole": Rule("xt.Multipole"),
    "Bend": Rule("xt.Bend"),
    "Solenoid": Rule("xt.UniformSolenoid"),
    "RFCavity": Rule("xt.Cavity", "EQUIVALENT", "CONST_P0",
                     "xtrack keeps p0c constant across RF: the gain lands in delta, "
                     "not in the reference momentum"),
    "FieldMap": Rule("xt.Cavity / xt.Drift", "EQUIVALENT", "FM_AS_CAVITY",
                     "field map replaced by an equivalent thick cavity (drift when no "
                     "effective voltage is known)"),
    "NCells": Rule("xt.Drift", "LOSSY", "NCELLS_TO_DRIFT",
                   "NCELLS cell train replaced by a drift of the same length"),
    "RFQCell": Rule("xt.Drift", "LOSSY", "RFQ_TO_DRIFT",
                    "RFQ cell replaced by a drift of the same length"),
    "Kicker": Rule("xt.Multipole"),
    "Collimator": Rule("xt.LimitRect/xt.LimitEllipse"),
    "Marker": Rule("xt.Marker"),
    "Instrument": Rule("xt.Drift/xt.Marker", "EQUIVALENT", "MONITOR_AS_DRIFT",
                       "xtrack has no beam-instrument element; written as a drift of the "
                       "same length (a marker when thin)"),
    "Foil": Rule("xt.Marker", "LOSSY", "FOIL_TO_MARKER",
                 "xtrack has no stripping foil; written as a marker"),
    "Taylor": Rule("xt.FirstOrderTaylorMap", "EQUIVALENT", "TAYLOR_BASIS_XTRACK",
                   "the matrix is re-used verbatim in xtrack's (x, px, y, py, zeta, delta) "
                   "basis; a source in another basis is not transformed"),
    "Patch": Rule("xt.XYShift/xt.SRotation/xt.XRotation/xt.YRotation"),
    "ReferenceChange": Rule("xt.ReferenceEnergyIncrease/xt.ZetaShift", "EQUIVALENT",
                            "REFCHANGE_AS_P0C",
                            "reference energy change written as a p0c increase "
                            "(xt.ReferenceEnergyIncrease)"),
    "Freq": Rule("xt.Marker", "EXACT", "OK",
                 "the RF clock lives on each xt.Cavity's frequency attribute"),
    "Directive": Rule("xt.Marker", "DROPPED", "FOREIGN_DIRECTIVE",
                      "format-specific directive has no xtrack form; written as a marker"),
    "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                          "overlapping fields written as consecutive elements"),
}

_MISSING = set(ALL_KINDS) - set(RULES)
if _MISSING:                                        # pragma: no cover - guarded by a test too
    raise RuntimeError(f"xtrack RULES do not cover {sorted(_MISSING)}")


# ---------------------------------------------------------------------------
# phase conventions
# ---------------------------------------------------------------------------
def xtrack_lag_deg(phase_rad: float) -> float:
    """IR synchronous phase [rad] → xtrack ``Cavity.lag`` [deg] (= φ·180/π + 90)."""
    return math.degrees(phase_rad) + 90.0


def xtrack_phase_rad(phase_rad: float) -> float:
    """IR synchronous phase [rad] → xtrack ``Cavity.phase`` [rad] (= φ + π/2)."""
    return phase_rad + math.pi / 2.0


def phase_from_xtrack(lag_deg: float = 0.0, phase_rad: float = 0.0) -> float:
    """xtrack ``Cavity`` (``lag`` deg + ``phase`` rad, they add) → IR synchronous phase."""
    from lattix.ir.units import wrap_rad

    return wrap_rad(math.radians(lag_deg) + phase_rad - math.pi / 2.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rot_pairs(el, shift: BodyShiftP | None) -> dict[str, float]:
    """Misalignment attributes for an xtrack element that has them (MAD-X EALIGN)."""
    if shift is None or shift.is_zero():
        return {}
    out: dict[str, float] = {}
    for attr, value in (("shift_x", shift.x_offset), ("shift_y", shift.y_offset),
                        ("shift_s", shift.z_offset), ("rot_s_rad_no_frame", shift.tilt),
                        ("rot_x_rad", shift.x_rot), ("rot_y_rad", shift.y_rot)):
        if value:
            out[attr] = float(value)
    return out


def _has_shift_fields(cls) -> bool:
    return "shift_x" in (getattr(cls, "_xofields", None) or {})


def _aperture_elements(ap: ApertureP, name: str, rep: FidelityReport, kind: str):
    """``ApertureP`` → one xtrack limit element (``LimitRect`` or ``LimitEllipse``)."""
    allow_jit()
    import xtrack as xt

    xl = ap.x_limits
    yl = ap.y_limits
    if xl is None and yl is None:
        return None
    if ap.shape == "RECTANGULAR":
        big = 1.0
        xmin, xmax = xl if xl is not None else (-big, big)
        ymin, ymax = yl if yl is not None else (-big, big)
        return xt.LimitRect(min_x=xmin, max_x=xmax, min_y=ymin, max_y=ymax)
    hx = ap.half_x if ap.half_x is not None else ap.half_y
    hy = ap.half_y if ap.half_y is not None else ap.half_x
    for lim in (xl, yl):
        if lim is not None and abs(lim[0] + lim[1]) > 1e-12 * max(1.0, abs(lim[1])):
            rep.lossy("APERTURE_OFFSET_DROPPED",
                      "xt.LimitEllipse is centred on the reference orbit; the aperture's "
                      "offset was dropped", element=name, kind=kind, limits=list(lim))
            break
    return xt.LimitEllipse(a=abs(hx), b=abs(hy))


def _species_name(mass_eV: float, charge: int) -> str:
    for nm, sp in SPECIES.items():
        if sp.charge == charge and abs(sp.mass_eV - mass_eV) <= 1e-4 * sp.mass_eV:
            return nm
    return ""


# ---------------------------------------------------------------------------
# IR -> xt.Line
# ---------------------------------------------------------------------------
class _Builder:
    """Accumulates ``(name, xt element, provenance row)`` triples in flat order."""

    def __init__(self, rep: FidelityReport) -> None:
        self.names = NameMap()
        self.elements: dict[str, Any] = {}
        self.order: list[str] = []
        self.rows: dict[str, dict] = {}
        self.rep = rep

    def add(self, base_name: str, el, *, ir: Element | None, role: str = "main",
            group: int | None = None) -> str:
        nm = self.names.assign(base_name)
        self.elements[nm] = el
        self.order.append(nm)
        row = {"role": role}
        if ir is not None:
            row["name"] = ir.name
            row["kind"] = ir.kind
            if ir.kind == "Instrument":
                row["family"] = ir.family
                if ir.params:
                    row["params"] = dict(ir.params)
            elif ir.kind == "Directive":
                row.update({"format": ir.format, "card": ir.card, "args": list(ir.args), "ir_role": ir.role})
            elif ir.kind == "Freq":
                row["frequency_Hz"] = ir.frequency_Hz
            elif ir.kind == "Foil":
                row.update({"material": ir.material, "thickness_kg_per_m2": ir.thickness_kg_per_m2,
                            "dE_ref_eV": ir.dE_ref_eV})
            elif ir.kind in ("RFCavity", "FieldMap") and getattr(ir, "rf", None) is not None:
                row["phase_rad"] = float(ir.rf.phase_rad)
                if ir.rf.dE_ref_eV is not None:
                    row["dE_ref_eV"] = float(ir.rf.dE_ref_eV)
                if ir.kind == "FieldMap":
                    # the row describes what the map became (the reader cannot give a map back):
                    # a cavity at the integrated (V_c, φs) or a drift
                    summary = (ir.meta or {}).get("map_summary") or {}
                    if summary.get("kind") == "rf" and summary.get("v_c_V"):
                        row["kind"] = "RFCavity"
                        row["phase_rad"] = float(summary.get("phase_sync_rad") or 0.0)
                        row["dE_ref_eV"] = float(summary.get("dE_ref_eV") or 0.0)
                    elif type(el).__name__ in ("Drift", "DriftExact"):
                        row["kind"] = "Drift"
                        row.pop("phase_rad", None)
                        row.pop("dE_ref_eV", None)
                    else:
                        row["kind"] = "RFCavity"
            if ir.provenance is not None:
                if ir.provenance.original_name:
                    row["original_name"] = ir.provenance.original_name
                if ir.provenance.original_type:
                    row["original_type"] = ir.provenance.original_type
                row["source_format"] = ir.provenance.format
        if group is not None:
            row["group"] = group
        self.rows[nm] = row
        return nm


def to_line(lattice: Lattice, *, energy_mode: str = "delta", report: FidelityReport | None = None,
            name: str | None = None, strict: bool = False, install_apertures: bool = True,
            bend_model: str | None = None, edge_model: str | None = None, rbend: bool = False):
    """Build an :class:`xtrack.Line` from an IR lattice, in flat order.

    Parameters
    ----------
    energy_mode:
        ``"delta"`` (default) normalizes every strength with the rigidity xtrack's own
        reference particle has there (see the module docstring); ``"local"`` uses Bρ at
        that element's entrance, ``"constant"`` the lattice start.  Either way the choice
        is recorded in the fidelity ledger (xtrack keeps p0c fixed through RF).
    report:
        optional :class:`~lattix.fidelity.FidelityReport` to fill; a fresh one is used
        when omitted (retrieve it through :class:`lattix.formats.xtrack.Writer`).
    install_apertures:
        emit ``LimitRect``/``LimitEllipse`` elements for :class:`ApertureP` data
        attached to ordinary elements (a :class:`Collimator` always becomes one).
    bend_model, edge_model, rbend:
        xtrack's ``Bend.model`` (``adaptive | full | bend-kick-bend | rot-kick-rot | mat-kick-mat |
        …``) and ``edge_entry_model``/``edge_exit_model`` (``linear | full | dipole-only |
        suppressed``) passed through verbatim (``BEND_MODEL_OPTION`` in the ledger); ``rbend``
        writes rectangular sources as ``xt.RBend`` instead of the default sector ``Bend``.
    """
    allow_jit()
    import xtrack as xt

    check_mode(energy_mode)
    rep = report if report is not None else FidelityReport()
    rep.target_format = "xtrack"

    placed = propagate(lattice)
    start_brho = lattice.reference.brho_signed
    probes = probe_momentum_ratio(placed, lattice.reference)
    b = _Builder(rep)
    b.bend_options = {k: v for k, v in (("model", bend_model), ("edge_entry_model", edge_model),
                                        ("edge_exit_model", edge_model)) if v}
    b.rbend = bool(rbend)

    for group, (p, probe) in enumerate(zip(placed, probes, strict=True)):
        el = p.element
        ref: ReferenceParticle = p.ref_in or lattice.reference
        brho = rigidity_for(energy_mode, ref.brho_signed, start_brho, probe)
        ratio = mode_ratio(energy_mode, ref.brho_signed, start_brho, probe)
        slip = 0.0
        if energy_mode == "delta" and el.kind in ("RFCavity", "FieldMap") and p.ref_in is not None:
            rf = getattr(el, "rf", None)
            f = (rf.frequency_Hz if rf is not None and rf.frequency_Hz else None) or ref.rf_frequency_Hz
            slip = phase_slip_turns(p.ref_in, p.s_in, el.length, f, lattice.reference)
            if slip_is_zero(slip):
                slip = 0.0                             # roundoff on a cavity at the start velocity
        _emit(b, el, brho, ref, lattice, group, rep, install_apertures, ratio=ratio, slip=slip,
              bend_options=b.bend_options, rbend=b.rbend)

    line = xt.Line(elements=b.elements, element_names=list(b.order))
    _attach_knobs(line, lattice, b, rep)
    sp = lattice.reference.species
    line.particle_ref = xt.Particles(mass0=sp.mass_eV, q0=sp.charge,
                                     kinetic_energy0=lattice.reference.kinetic_energy_eV)
    meta = {
        "version": __version__,
        "source_format": lattice.meta.get("source_format", "IR"),
        "lattice": lattice.name,
        "use": lattice.use,
        "energy_mode": energy_mode,
        "species": sp.name,
        "mass_eV": sp.mass_eV,
        "charge": sp.charge,
        "kinetic_energy_eV": lattice.reference.kinetic_energy_eV,
        "elements": b.rows,
    }
    if lattice.reference.rf_frequency_Hz:
        meta["rf_frequency_Hz"] = lattice.reference.rf_frequency_Hz
    line.metadata = {METADATA_KEY: meta}

    _record_energy_mode(placed, energy_mode, rep)
    rep.raise_if(strict)
    if name:
        line.name = name
    return line


def to_environment(lattice: Lattice, *, energy_mode: str = "delta", report: FidelityReport | None = None,
                   strict: bool = False, install_apertures: bool = True):
    """An :class:`xtrack.Environment` whose lines mirror the IR's ``lines`` (nested, with repeats
    expanded and reversed sub-lines written out), built on the flat :func:`to_line` conversion:
    every IR definition becomes one xtrack element (its first placement's conversion — a
    definition used at two reference energies keeps the first, ``MULTI_RIGIDITY_DEFINITION``),
    and every IR line becomes ``env.new_line``.  Knobs come along."""
    allow_jit()
    import xtrack as xt

    rep = report if report is not None else FidelityReport()
    line = to_line(lattice, energy_mode=energy_mode, report=rep, install_apertures=install_apertures)
    flat = list(lattice.flatten())
    # first xtrack element per IR definition (the flat line has one xtrack element per placement)
    rows = line.metadata[METADATA_KEY]["elements"]
    main_names = [nm for nm in line.element_names if rows[nm].get("role", "main") == "main"]
    by_ir: dict[str, str] = {}
    for placed_el, nm in zip(flat, main_names, strict=False):
        ir_name = placed_el.element.name
        if ir_name not in by_ir:
            by_ir[ir_name] = nm
        elif line.element_dict[nm].to_dict() != line.element_dict[by_ir[ir_name]].to_dict():
            rep.equivalent("MULTI_RIGIDITY_DEFINITION",
                           "one definition is used at two reference energies (or arrival times); the first "
                           "placement's conversion is the environment's element",
                           element=ir_name, kind=placed_el.element.kind)
    keep = {nm: line.element_dict[nm] for nm in set(by_ir.values())}
    # sibling elements of a placement (apertures, split drifts) are placement-specific: only the
    # main element is shared, the siblings of the first placement follow it in a private sub-line
    siblings: dict[str, list[str]] = {}
    names_all = list(line.element_names)
    for nm in by_ir.values():
        i = names_all.index(nm)
        group = line.metadata[METADATA_KEY]["elements"][nm].get("group")
        sibs = []
        for j in (i - 1, i + 1, i + 2):
            if 0 <= j < len(names_all):
                r = line.metadata[METADATA_KEY]["elements"][names_all[j]]
                if r.get("group") == group and r.get("role", "main") != "main":
                    sibs.append((j, names_all[j]))
        for _j, s in sibs:
            keep[s] = line.element_dict[s]
        siblings[nm] = [s for _j, s in sorted(sibs)]
    env = xt.Environment(element_dict=keep, particle_ref=line.particle_ref)
    for n in list(getattr(line, "vars", {}).keys()) if lattice.variables else []:
        if n in ("t_turn_s", "__vary_default") or n.startswith("__"):
            continue
        env.vars[n] = float(line.vars[n]._value)
    for n, var in lattice.variables.items():
        if var.expression is not None:
            try:
                env.vars[n] = env._xdeps_eval.eval(var.expression.text)
            except Exception:                            # noqa: BLE001 - already noted by to_line
                pass
    # lines in dependency order
    done: set[str] = set()

    def components_of(ir_line) -> list[str]:
        comps: list[str] = []
        for it in ir_line.items:
            for _ in range(max(1, it.repeat)):
                if it.ref in lattice.lines:
                    if it.ref not in done:
                        emit(it.ref)
                    comps.append(it.ref if not it.reverse else f"{it.ref}_reversed")
                    if it.reverse and f"{it.ref}_reversed" not in done:
                        rev = list(reversed(env.lines[it.ref].element_names))
                        env.new_line(name=f"{it.ref}_reversed", components=rev)
                        done.add(f"{it.ref}_reversed")
                        rep.equivalent("REVERSED_LINE_EXPANDED", "a reversed sub-line is written element by "
                                       "element in reverse order (asymmetric elements are not flipped)",
                                       element=None, kind=None, sub_line=it.ref)
                else:
                    nm = by_ir.get(it.ref)
                    if nm is None:
                        continue
                    pre = [s for s in siblings.get(nm, []) if names_all.index(s) < names_all.index(nm)]
                    post = [s for s in siblings.get(nm, []) if names_all.index(s) > names_all.index(nm)]
                    comps += pre + [nm] + post
        return comps

    def emit(name: str) -> None:
        done.add(name)
        env.new_line(name=name, components=components_of(lattice.lines[name]))

    root = lattice.use or (next(iter(lattice.lines)) if lattice.lines else None)
    for name in lattice.lines:
        if name not in done:
            emit(name)
    meta = dict(line.metadata)
    meta[METADATA_KEY] = dict(meta[METADATA_KEY])
    meta[METADATA_KEY]["use"] = root
    meta[METADATA_KEY]["document"] = "environment"
    env.metadata = meta
    for name in lattice.lines:
        if name in env.lines:
            env.lines[name].metadata = {METADATA_KEY: dict(meta[METADATA_KEY], use=name, lattice=lattice.name)}
    rep.raise_if(strict)
    return env


def _record_energy_mode(placed: list[Placed], energy_mode: str, rep: FidelityReport) -> None:
    for p in placed:
        ref = p.ref_in
        if ref is None:
            continue
        dE = energy_gain_eV(p.element, ref)
        if abs(dE) <= _ACCEL_TOL_eV:
            continue
        record_rigidity_mode(rep, energy_mode, element=p.element.name, kind=p.element.kind,
                             dE_eV=dE, brho=ref.brho_signed)


def _record(rep: FidelityReport, el: Element, rule: Rule, **details) -> None:
    if rule.cls == "EXACT":
        rep.exact(el.name, el.kind)
    else:
        rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind,
                target=rule.target, **details)


def _emit(b: _Builder, el: Element, brho: float, ref: ReferenceParticle, lattice: Lattice,
          group: int, rep: FidelityReport, install_apertures: bool, *, ratio: float = 1.0,
          slip: float = 0.0, bend_options: dict | None = None, rbend: bool = False) -> None:
    rule = RULES.get(el.kind)
    if rule is None:                                # pragma: no cover - RULES is total
        raise KeyError(f"xtrack writer has no rule for kind {el.kind!r}")

    if isinstance(el, Superposition):
        rep.lossy("SUPERPOSITION_FLATTENED",
                  "overlapping fields written as consecutive xtrack elements",
                  element=el.name, kind="Superposition", children=len(el.children))
        for _offset, child_name in el.children:
            child = lattice.elements.get(child_name)
            if child is None:
                rep.dropped("SUPERPOSITION_CHILD_MISSING",
                            f"superposition child {child_name!r} is not defined",
                            element=el.name, kind="Superposition")
                continue
            _emit(b, child, brho, ref, lattice, group, rep, install_apertures, ratio=ratio, slip=slip,
                  bend_options=bend_options, rbend=rbend)
        return

    if isinstance(el, Patch):
        _emit_patch(b, el, group, rep)
        _record(rep, el, rule)
        return

    made = _build(el, brho, ref, rep, ratio=ratio, slip=slip, bend_options=bend_options, rbend=rbend)
    if made is None:                                # pragma: no cover - _build is total
        raise KeyError(f"xtrack writer produced nothing for {el.kind!r}")
    main, extra = made

    # -- misalignment ------------------------------------------------------
    shift = el.shift
    if shift is not None and not shift.is_zero():
        if _has_shift_fields(type(main)):
            for attr, value in _rot_pairs(main, shift).items():
                setattr(main, attr, value)
        else:
            rep.equivalent("SHIFT_NO_OP",
                           f"xt.{type(main).__name__} is invariant under a transverse shift "
                           "or roll; the body shift changes no physics and was not emitted",
                           element=el.name, kind=el.kind,
                           x_offset=shift.x_offset, y_offset=shift.y_offset, tilt=shift.tilt)

    # -- apertures ---------------------------------------------------------
    ap = el.aperture
    pre: list = []
    post: list = []
    if install_apertures and ap is not None and not isinstance(el, Collimator):
        lim = _aperture_elements(ap, el.name, rep, el.kind)
        if lim is not None:
            if ap.aperture_at in ("ENTRANCE", "BOTH_ENDS", "CONTINUOUS"):
                pre.append(lim)
            if ap.aperture_at in ("EXIT", "BOTH_ENDS", "CONTINUOUS"):
                post.append(lim.copy() if pre else lim)
            if ap.aperture_at == "CONTINUOUS":
                rep.lossy("APERTURE_CONTINUOUS_AT_ENDS",
                          "a continuous aperture is checked only at the element ends in xtrack",
                          element=el.name, kind=el.kind)

    for i, lim in enumerate(pre):
        b.add(f"{el.name}_aper_in" if i == 0 else f"{el.name}_aper_in{i}", lim, ir=el,
              role="aperture_entry", group=group)
    b.add(el.name, main, ir=el, role="main", group=group)
    for i, x in enumerate(extra):
        b.add(f"{el.name}_x{i}", x, ir=el, role="extra", group=group)
    for i, lim in enumerate(post):
        b.add(f"{el.name}_aper_out" if i == 0 else f"{el.name}_aper_out{i}", lim, ir=el,
              role="aperture_exit", group=group)

    if isinstance(el, Directive) and el.role in COMMENT_ROLES:
        rep.exact(el.name, "Directive", code="DIRECTIVE_AS_MARKER",
                  message=f"role {el.role!r} carries no xtrack physics; written as a marker")
    else:
        _record(rep, el, rule)
    if isinstance(el, (RFCavity, FieldMap)):
        dE = energy_gain_eV(el, ref)
        if abs(dE) > _ACCEL_TOL_eV and el.kind != "RFCavity":
            rep.equivalent("CONST_P0",
                           "xtrack keeps p0c constant across RF: the reference momentum does "
                           "not follow this element's gain",
                           element=el.name, kind=el.kind, dE_eV=dE)


def _emit_patch(b: _Builder, el: Patch, group: int, rep: FidelityReport) -> None:
    """A ``Patch`` becomes the xtrack frame elements it is made of, in survey order."""
    allow_jit()
    import xtrack as xt

    n = 0
    if el.x_offset or el.y_offset:
        b.add(f"{el.name}_xy", _new(xt.XYShift, dx=el.x_offset, dy=el.y_offset), ir=el,
              role="patch", group=group)
        n += 1
    if el.y_rot:
        b.add(f"{el.name}_yrot", _make_rotation(xt.YRotation, el.y_rot), ir=el,
              role="patch", group=group)
        n += 1
    if el.x_rot:
        b.add(f"{el.name}_xrot", _make_rotation(xt.XRotation, el.x_rot), ir=el,
              role="patch", group=group)
        n += 1
    if el.tilt:
        b.add(f"{el.name}_srot", _make_rotation(xt.SRotation, el.tilt), ir=el,
              role="patch", group=group)
        n += 1
    if el.z_offset or el.length:
        b.add(f"{el.name}_ds", xt.Drift(length=el.z_offset + el.length), ir=el,
              role="patch", group=group)
        n += 1
    if el.t_offset_s:
        rep.lossy("PATCH_T_OFFSET_DROPPED",
                  "xtrack has no reference-time patch on a frame element",
                  element=el.name, kind="Patch", t_offset_s=el.t_offset_s)
    if el.e_tot_offset_eV:
        rep.lossy("PATCH_E_OFFSET_DROPPED",
                  "a patch energy offset is not representable; use a ReferenceChange",
                  element=el.name, kind="Patch", e_tot_offset_eV=el.e_tot_offset_eV)
    if n == 0:
        b.add(el.name, xt.Marker(), ir=el, role="main", group=group)


def _new(cls, **kw):
    """Construct an xtrack element, muting the 0.112 ``FutureWarning``s for the frame
    classes: ``XYShift``/``ZetaShift``/``SRotation``/``XRotation``/``YRotation`` are
    deprecated there in favour of ``Translation``/``TimeDelay``/``Rotation``, none of
    which exist in 0.103.5, so the old spellings are the only ones that serve both."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return cls(**kw)


def _make_rotation(cls, angle_rad: float):
    """``SRotation``/``XRotation``/``YRotation`` take **degrees**."""
    return _new(cls, angle=math.degrees(angle_rad))


def _cavity(el, brho, ref, rep, *, voltage: float, length: float, slip: float = 0.0,
            phase_rad: float | None = None):
    allow_jit()
    import xtrack as xt

    rf = el.rf
    phi = rf.phase_rad if phase_rad is None else phase_rad
    kw: dict[str, Any] = {"voltage": voltage, "phase": xtrack_phase_rad(phi) - 2.0 * math.pi * slip}
    if not slip_is_zero(slip):
        record_phase_slip(rep, element=el.name, kind=el.kind, slip_turns=slip, engine="xtrack", attribute="phase")
    if rf.frequency_Hz:
        kw["frequency"] = rf.frequency_Hz
    elif ref.rf_frequency_Hz:
        kw["frequency"] = ref.rf_frequency_Hz
        rep.equivalent("RF_FREQUENCY_FROM_CLOCK",
                       "the element carried no frequency; the reference RF clock was used",
                       element=el.name, kind=el.kind, frequency_Hz=ref.rf_frequency_Hz)
    elif rf.harmon:
        rep.equivalent("RF_FREQUENCY_FROM_HARMONIC",
                       "the deck gives a harmonic number instead of a frequency; xt.Cavity "
                       "keeps the harmonic and its frequency stays 0 Hz until a twiss sets it",
                       element=el.name, kind=el.kind, harmon=rf.harmon)
    else:
        rep.lossy("RF_FREQUENCY_MISSING",
                  "no RF frequency is known; xt.Cavity defaults to 0 Hz (no phase slip)",
                  element=el.name, kind=el.kind)
    if rf.harmon:
        kw["harmonic"] = int(rf.harmon)
    if length:
        kw["length"] = length
    return xt.Cavity(**kw)


def _build(el: Element, brho: float, ref: ReferenceParticle, rep: FidelityReport, *, ratio: float = 1.0,
           slip: float = 0.0, bend_options: dict | None = None, rbend: bool = False):
    """``(main xtrack element, [extra elements])`` for one IR element.  ``ratio`` =
    Bρ_local/Bρ_used (energy mode) rescales the quantities that are not normalized strengths
    (kicks, maps, a bend's k0); ``slip`` [turns] is the constant-velocity clock's phase slip a
    cavity is written against."""
    allow_jit()
    import xtrack as xt

    # a passthrough recorded by from_line for a class the IR cannot model
    raw = (el.native.get("xtrack") or {}).get("element")
    if raw is not None and isinstance(el, Marker):
        rep.equivalent("NATIVE_PASSTHROUGH",
                       "an xtrack element class the IR cannot model was re-emitted verbatim "
                       "from native['xtrack']", element=el.name, kind="Marker",
                       xtrack_class=raw.get("__class__"))
        return _from_element_dict(dict(raw)), []

    if isinstance(el, Drift):
        return xt.Drift(length=el.length), []

    if isinstance(el, Quadrupole):
        note_quad_higher_orders(el, rep, "xtrack")
        m = el.multipole
        kw = {"length": el.length, "k1": m.Bn.get(1, 0.0) / brho}
        if m.Bs.get(1):
            kw["k1s"] = m.Bs[1] / brho
        if m.tilt.get(1):
            kw["rot_s_rad"] = m.tilt[1]
        return xt.Quadrupole(**kw), []

    if isinstance(el, Sextupole):
        m = el.multipole
        kw = {"length": el.length, "k2": m.Bn.get(2, 0.0) / brho}
        if m.Bs.get(2):
            kw["k2s"] = m.Bs[2] / brho
        if m.tilt.get(2):
            kw["rot_s_rad"] = m.tilt[2]
        return xt.Sextupole(**kw), []

    if isinstance(el, Octupole):
        m = el.multipole
        kw = {"length": el.length, "k3": m.Bn.get(3, 0.0) / brho}
        if m.Bs.get(3):
            kw["k3s"] = m.Bs[3] / brho
        if m.tilt.get(3):
            kw["rot_s_rad"] = m.tilt[3]
        return xt.Octupole(**kw), []

    if isinstance(el, Multipole):
        m = el.multipole
        order = max([0] + list(m.BnL) + list(m.BsL))
        knl = [m.BnL.get(i, 0.0) / brho for i in range(order + 1)]
        ksl = [m.BsL.get(i, 0.0) / brho for i in range(order + 1)]
        native = el.native.get("xtrack") or {}
        lrad = native.get("lrad") or 0.0
        kw: dict[str, Any] = {"knl": knl, "ksl": ksl, "length": el.length or lrad}
        if el.length:
            # measured: xt.Multipole only advances s when isthick is set (this is how
            # xt.Line.from_madx_sequence imports a thick MAD-X kicker/multipole; the kick
            # then sits at the centre of a drift of the same length)
            kw["isthick"] = True
        if native.get("hxl"):
            kw["hxl"] = float(native["hxl"])
        if m.tilt.get(0):
            kw["rot_s_rad"] = m.tilt[0]
        return xt.Multipole(**kw), []

    if isinstance(el, Bend):
        bd = el.bend
        kw: dict[str, Any] = {"length": el.length, "angle": bd.angle}
        if bd.e1:
            kw["edge_entry_angle"] = bd.e1
        if bd.e2:
            kw["edge_exit_angle"] = bd.e2
        if bd.edge_int1:
            kw["edge_entry_fint"] = bd.edge_int1
        kw["edge_exit_fint"] = bd.edge_int1 if bd.edge_int2 is None else bd.edge_int2
        if bd.hgap:
            kw["edge_entry_hgap"] = bd.hgap
            kw["edge_exit_hgap"] = bd.hgap
        if el.multipole.Bn.get(1):
            kw["k1"] = el.multipole.Bn[1] / brho
        if el.multipole.Bn.get(2):
            kw["k2"] = el.multipole.Bn[2] / brho
        if bd.tilt_ref:
            kw["rot_s_rad"] = bd.tilt_ref
        if not el.length and bd.angle:
            rep.lossy("THIN_BEND_AS_MULTIPOLE",
                      "a zero-length bend has no xt.Bend form; written as a thin multipole "
                      "with hxl", element=el.name, kind="Bend", angle=bd.angle)
            return xt.Multipole(knl=[bd.angle], ksl=[0.0], hxl=bd.angle, length=0.0), []
        k0 = (el.native.get("xtrack") or {}).get("k0")
        if k0 is not None:
            kw["k0"] = float(k0)
        elif abs(ratio - 1.0) > 1e-15 and el.length:
            # xtrack's reference particle carries the RF gain as delta: a field k0 = h·r bends
            # it by exactly the design angle (h alone would under-bend it by 1/r)
            kw["k0"] = ratio * bd.angle / el.length
        if bend_options:
            kw.update(bend_options)
            rep.equivalent("BEND_MODEL_OPTION", "xtrack bend/edge model options written as requested "
                           f"({', '.join(f'{k}={v}' for k, v in bend_options.items())})",
                           element=el.name, kind="Bend")
        if rbend and bd.rect and el.length and bd.angle:
            # xt.RBend takes the straight length and its own e1/e2 relative to the rectangular faces
            half = bd.angle / 2.0
            kw_r = dict(kw)
            kw_r.pop("length", None)
            kw_r["length_straight"] = el.length * math.sin(half) / half
            kw_r["edge_entry_angle"] = bd.e1 - half
            kw_r["edge_exit_angle"] = bd.e2 - half
            rep.equivalent("RBEND_WRITTEN", "a rectangular source bend written as xt.RBend (straight length, "
                           "face angles relative to the rectangular faces)", element=el.name, kind="Bend")
            return xt.RBend(**kw_r), []
        return xt.Bend(**kw), []

    if isinstance(el, Solenoid):
        ks = el.solenoid.Bsol_T / brho
        if not el.length:
            rep.lossy("THIN_SOLENOID_DROPPED",
                      "xt.UniformSolenoid needs a length; a zero-length solenoid was "
                      "written as a marker", element=el.name, kind="Solenoid", ks=ks)
            return xt.Marker(), []
        return xt.UniformSolenoid(length=el.length, ks=ks), []

    if isinstance(el, RFCavity):
        rf = el.rf
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        return _cavity(el, brho, ref, rep, voltage=volt, length=el.length, slip=slip), []

    if isinstance(el, FieldMap):
        rf = el.rf
        summary = (el.meta or {}).get("map_summary") or {}
        if summary.get("kind") == "rf" and summary.get("v_c_V"):
            # (V_c, φs) of the integrated map: dE_ref = V_c·cos φs by construction — the card phase
            # of a relative-phase map is *not* its synchronous phase (lattix.ir.fieldmap._cavity_numbers)
            return _cavity(el, brho, ref, rep, voltage=float(summary["v_c_V"]), length=el.length, slip=slip,
                           phase_rad=float(summary.get("phase_sync_rad") or 0.0)), []
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        if not volt and rf.dE_ref_eV:
            volt = rf.dE_ref_eV / max(math.cos(rf.phase_rad), 1e-12)
        if volt:
            return _cavity(el, brho, ref, rep, voltage=volt, length=el.length, slip=slip), []
        rep.lossy("FM_TO_DRIFT",
                  "field map has no effective voltage; replaced by a drift of the same length",
                  element=el.name, kind="FieldMap")
        return xt.Drift(length=el.length), []

    if isinstance(el, (NCells, RFQCell)):
        return xt.Drift(length=el.length), []

    if isinstance(el, Kicker):
        if el.electric:
            rep.lossy("EKICK_AS_MAGNETIC",
                      "electric steerer written as a magnetic thin multipole",
                      element=el.name, kind="Kicker")
        # measured: knl[0] = -hkick, ksl[0] = +vkick reproduces px += hkick, py += vkick;
        # a thick kicker needs isthick=True to advance s (what from_madx_sequence does)
        kw = {"knl": [-el.hkick * ratio], "ksl": [el.vkick * ratio], "length": el.length}
        if el.length:
            kw["isthick"] = True
        return xt.Multipole(**kw), []

    if isinstance(el, Collimator):
        ap = el.aperture
        lim = _aperture_elements(ap, el.name, rep, "Collimator") if ap is not None else None
        if lim is None:
            rep.lossy("COLLIMATOR_WITHOUT_APERTURE",
                      "collimator carries no aperture; written as a drift/marker",
                      element=el.name, kind="Collimator")
            return (xt.Drift(length=el.length) if el.length else xt.Marker()), []
        extra = [xt.Drift(length=el.length)] if el.length else []
        return lim, extra

    if isinstance(el, Instrument):
        return (xt.Drift(length=el.length) if el.length else xt.Marker()), []

    if isinstance(el, (Marker, Foil, Freq, Directive)):
        return xt.Marker(), []

    if isinstance(el, Taylor):
        import numpy as np

        payload = (el.native.get("xtrack") or {}).get("element")
        if isinstance(payload, dict) and payload.get("__class__") in ("DipoleEdge", "SecondOrderTaylorMap"):
            rep.equivalent("NATIVE_PASSTHROUGH",
                           f"the {payload['__class__']} this map was read from is re-emitted verbatim",
                           element=el.name, kind="Taylor", xtrack_class=payload["__class__"])
            return _from_element_dict(dict(payload)), []
        matrix, offset = scale_taylor(el.matrix, el.offset, ratio)
        return xt.FirstOrderTaylorMap(length=el.length,
                                      m0=np.asarray(offset, dtype=float),
                                      m1=np.asarray(matrix, dtype=float)), []

    if isinstance(el, ReferenceChange):
        dE = energy_gain_eV(el, ref)
        out = []
        if dE:
            p_in = ref.pc_eV
            p_out = ref.advanced(dE_eV=dE).pc_eV
            out.append(xt.ReferenceEnergyIncrease(Delta_p0c=p_out - p_in))
        dt = el.dtime_s
        if not dt and el.dphase_rad and ref.rf_frequency_Hz:
            dt = el.dphase_rad / (2.0 * math.pi * ref.rf_frequency_Hz)
        if dt:
            # measured: ZetaShift maps zeta -> zeta - dzeta; a later reference clock
            # (dt > 0) puts every particle ahead by +beta0*c*dt
            out.append(_new(xt.ZetaShift, dzeta=-ref.beta * C_LIGHT * dt))
        if not out:
            return xt.Marker(), []
        return out[0], out[1:]

    raise KeyError(f"xtrack writer cannot build {el.kind!r}")   # pragma: no cover


def _from_element_dict(d: dict):
    """Rebuild an xtrack element from its ``to_dict`` payload."""
    allow_jit()
    import xtrack as xt

    cls = getattr(xt, d.pop("__class__"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# xt.Line -> IR
# ---------------------------------------------------------------------------
#: :func:`_convert_element` maps these xtrack classes onto IR kinds (the core ones here, the rest of
#: the zoo in :mod:`lattix.formats.xtrack.extra_elements`, slices through :func:`_unslice`); anything
#: else becomes a Marker with a ``DROPPED:UNSUPPORTED_XTRACK_ELEMENT`` entry and a verbatim
#: ``native`` passthrough that :func:`to_line` re-emits unchanged.
CORE_CLASSES = frozenset({
    "Drift", "DriftExact", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Bend",
    "RBend", "UniformSolenoid", "Solenoid", "Cavity", "Marker", "LimitRect", "LimitEllipse",
    "XYShift", "SRotation", "XRotation", "YRotation", "ZetaShift", "FirstOrderTaylorMap",
    "ReferenceEnergyIncrease",
})
from lattix.formats.xtrack.extra_elements import HANDLED_HERE as _EXTRA_CLASSES  # noqa: E402

KNOWN_CLASSES = CORE_CLASSES | _EXTRA_CLASSES
SLICE_PREFIXES = ("DriftSlice", "DriftExactSlice", "ThinSlice", "ThickSlice")


def is_known_class(cname: str) -> bool:
    """Every xtrack ``BeamElement`` class the reader maps, folds, drops by name or merges as a slice."""
    return cname in KNOWN_CLASSES or cname.startswith(SLICE_PREFIXES)


def _particle_ref_reference(line, warn: list[str]) -> ReferenceParticle | None:
    import numpy as np

    pref = getattr(line, "particle_ref", None)
    if pref is None:
        return None
    mass_eV = float(np.atleast_1d(pref.mass0)[0])
    charge = int(round(float(np.atleast_1d(pref.q0)[0])))
    ke = float(np.atleast_1d(pref.kinetic_energy0)[0])
    meta_ke = ((getattr(line, "metadata", None) or {}).get(METADATA_KEY) or {}).get("kinetic_energy_eV")
    if meta_ke is not None and abs(float(meta_ke) - ke) <= 1e-9 * max(1.0, abs(ke)):
        ke = float(meta_ke)                        # the value the writer had, before the p0c round trip
    nm = _species_name(mass_eV, charge)
    meta_sp = ((getattr(line, "metadata", None) or {}).get(METADATA_KEY) or {}).get("species")
    if not nm and isinstance(meta_sp, str) and meta_sp:
        # a custom species keeps the name the writer had (e.g. FLAME's ``ion_A238_Q33``)
        sp = Species(name=meta_sp, mass_eV=mass_eV, charge=charge)
    else:
        sp = SPECIES[nm] if nm else Species(name=f"q{charge}m{mass_eV:.6g}", mass_eV=mass_eV,
                                            charge=charge)
    if not nm:
        warn.append(f"particle_ref mass0={mass_eV} eV charge={charge} matches no known "
                    f"species; a custom Species was created")
    return ReferenceParticle(species=sp, kinetic_energy_eV=ke)


def _unslice(el) -> tuple[Any, float, str]:
    """``(element to convert, length weight, slice kind)`` for a sliced xtrack element."""
    parent = getattr(el, "_parent", None)
    if parent is None:
        return el, 1.0, ""
    weight = float(getattr(el, "weight", 1.0) or 0.0)
    cname = type(el).__name__
    if cname.startswith("Drift"):
        return None, weight, "drift"
    if cname.startswith("Thick"):
        return parent, weight, "thick"
    return parent, weight, "thin"


def _restore_kind(el: Element, row: dict) -> Element:
    """A marker/drift that the writer's metadata says was an Instrument or a Directive."""
    kind = row.get("kind")
    common = {"name": el.name, "length": el.length, "aperture": el.aperture, "shift": el.shift,
              "provenance": el.provenance, "meta": el.meta}
    if kind == "Instrument" and el.kind in ("Marker", "Drift"):
        return Instrument(family=row.get("family", "MONITOR"), params=dict(row.get("params") or {}), **common)
    if kind == "Directive" and el.kind == "Marker":
        return Directive(format=row.get("format", "tracewin"), card=row.get("card", ""),
                         args=list(row.get("args") or []), role=row.get("ir_role", "other"), **common)
    if kind == "Freq" and el.kind == "Marker" and row.get("frequency_Hz") is not None:
        return Freq(frequency_Hz=float(row["frequency_Hz"]), **common)
    if kind == "Collimator" and el.kind in ("Marker", "Drift") and el.aperture is None:
        return Collimator(**common)                   # a collimator without limits was written as a marker/drift
    if kind == "Foil" and el.kind == "Marker":
        return Foil(material=row.get("material", "C"), thickness_kg_per_m2=float(row.get("thickness_kg_per_m2") or 0.0),
                    dE_ref_eV=row.get("dE_ref_eV"), **common)
    return el


def _lattice_with_shared_definitions(name: str, out: list[Element], ref: ReferenceParticle,
                                     rows: list[dict], line_name: str | None = None) -> Lattice:
    """Like ``Lattice.from_sequence`` but two placements that the metadata traces to the same IR
    definition (same IR name, identical parameters) share one definition again, so
    write → read → write is a fixed point for lines that reuse an element."""
    lat = Lattice(name=name, reference=ref)
    line = Line(name=line_name or name)
    seen: dict[str, tuple[str, dict]] = {}
    by_ir = {}
    for el in out:
        ir_name = None
        for r in rows:
            if r.get("name") and r.get("role", "main") == "main" and r.get("kind") == el.kind and \
                    id(r) not in by_ir and (r.get("name") == el.name or el.name.startswith(r["name"])):
                ir_name = r["name"]
                by_ir[id(r)] = True
                break
        key = ir_name or el.name
        dump = el.model_dump(exclude={"name", "provenance"})
        if key in seen and seen[key][1] == dump:
            line.items.append(LineItem(ref=seen[key][0]))
            continue
        registered = lat.add_element(el)
        seen.setdefault(key, (registered, dump))
        line.items.append(LineItem(ref=registered))
    lat.lines[line.name] = line
    lat.use = line.name
    return lat


def from_line(line, reference: ReferenceParticle | None = None, *,
              report: FidelityReport | None = None, name: str | None = None,
              energy_mode: str | None = None, extra_lines: dict[str, list[str]] | None = None,
              root_components: list[str] | None = None) -> Lattice:
    """Convert an :class:`xtrack.Line` back to an IR :class:`~lattix.ir.lattice.Lattice`.

    ``reference`` wins over the line's ``particle_ref``; without either, a 1 GeV proton
    is assumed and a warning is recorded (normalized strengths are meaningless without
    a rigidity).  When the line carries ``metadata["lattix"]`` the original IR kinds,
    element names and energy mode are restored, so ``from_line(to_line(lat))`` is a
    fixed point.
    """
    rep = report if report is not None else FidelityReport()
    rep.source_format = "xtrack"
    warn: list[str] = []
    meta = dict(getattr(line, "metadata", None) or {}).get(METADATA_KEY) or {}
    rows: dict[str, dict] = meta.get("elements") or {}

    ref = reference or _particle_ref_reference(line, warn)
    if ref is None:
        ref = ReferenceParticle(species=SPECIES["proton"], kinetic_energy_eV=1e9)
        warn.append("the line carries no particle_ref and no reference was given; "
                    "normalized strengths were converted at 1 GeV protons")
    if meta.get("rf_frequency_Hz"):
        ref = ref.model_copy(update={"rf_frequency_Hz": float(meta["rf_frequency_Hz"])})

    names = list(line.element_names)
    elements = line.element_dict if hasattr(line, "element_dict") else line.elements
    out: list[Element] = []
    scaled: list[tuple[Element, dict]] = []      # (ir element, normalized strengths)
    skip_groups: set[int] = set()

    from lattix.formats.xtrack.extra_elements import fold_edge, misalignment_as_patch, misalignment_shift

    placed_x: list[tuple[str, Element]] = []           # (xtrack name, IR element) for the knob pass
    out_x: list[str | None] = []                       # xtrack name of each `out` entry (None: synthesized)
    pending_shift: tuple[str, BodyShiftP] | None = None  # an entry Misalignment waiting for its element
    pending_edge: tuple[Any, str, str] | None = None     # (raw, class, xname) entry edge waiting for a bend
    for xname in names:
        raw = elements[xname] if isinstance(elements, dict) else elements[names.index(xname)]
        row = rows.get(xname) or {}
        role = row.get("role", "main")
        group = row.get("group")
        if role in ("aperture_entry", "aperture_exit", "extra") and group is not None:
            continue                                     # folded into the main element
        if role == "patch" and group is not None:
            if group in skip_groups:
                continue
            skip_groups.add(group)
            patch = _patch_group(line, names, rows, group, elements, row)
            rep.exact(patch.name, "Patch")
            out.append(patch)
            out_x.append(None)
            continue
        cname = type(raw).__name__
        ir_name = row.get("name") or xname
        if cname == "Misalignment":
            if not bool(getattr(raw, "is_exit", False)):
                if pending_shift is not None and pending_shift[0] != "__consumed__":
                    out.append(misalignment_as_patch(elements[pending_shift[0]], pending_shift[0], rep))
                    out_x.append(None)
                pending_shift = (xname, misalignment_shift(raw))
                continue
            if pending_shift is None or pending_shift[0] != "__consumed__":
                out.append(misalignment_as_patch(raw, ir_name, rep))
                out_x.append(None)
            pending_shift = None
            continue
        if cname in ("DipoleEdge", "MagnetEdge", "DipoleFringe"):
            side = getattr(raw, "side", None)
            is_exit = bool(getattr(raw, "is_exit", False)) if cname == "MagnetEdge" else (str(side) == "exit")
            if cname == "DipoleFringe":
                is_exit = bool(out) and isinstance(out[-1], Bend)
            if not is_exit:
                if pending_edge is not None:
                    dangling = fold_edge(pending_edge[0], pending_edge[1], None, "entry", pending_edge[2], rep)
                    if dangling is not None:
                        out.append(dangling)
                        out_x.append(None)
                pending_edge = (raw, cname, ir_name)
                continue
            lens = fold_edge(raw, cname, out[-1] if out else None, "exit", ir_name, rep)
            if lens is not None:
                out.append(lens)
                out_x.append(None)
            continue

        el, norm = _convert_element(raw, xname, row, rep, warn)
        if el is None:
            continue
        el = _restore_kind(el, row)
        # aperture data and a thick collimator's drift live on sibling elements
        _attach_group_extras(el, names, rows, elements, group)
        if pending_edge is not None:
            dangling = fold_edge(pending_edge[0], pending_edge[1], el, "entry", pending_edge[2], rep)
            if dangling is not None:
                out.append(dangling)
                out_x.append(None)
            pending_edge = None
        if pending_shift is not None and pending_shift[0] != "__consumed__":
            if el.kind in ("Marker", "Patch", "ReferenceChange", "Freq", "Directive"):
                out.append(misalignment_as_patch(elements[pending_shift[0]], pending_shift[0], rep))
                out_x.append(None)
            else:
                el.shift = pending_shift[1]
                rep.equivalent("MISALIGNMENT_FOLDED", "an xt.Misalignment pair became the enclosed element's "
                               "body shift", element=el.name, kind=el.kind)
                anchor = float(getattr(elements[pending_shift[0]], "anchor", 0.0) or 0.0)
                if anchor and abs(anchor - 0.5 * el.length) > 1e-12:
                    el.native.setdefault("xtrack", {})["misalignment_anchor"] = anchor
                    rep.equivalent("MISALIGNMENT_ANCHOR", "the misalignment's rotation anchor is not the element "
                                   "centre; kept as a native passthrough", element=el.name, kind=el.kind,
                                   anchor=anchor)
            pending_shift = ("__consumed__", pending_shift[1])
        out.append(el)
        out_x.append(xname)
        placed_x.append((xname, el))
        scaled.append((el, norm))
    if pending_edge is not None:
        dangling = fold_edge(pending_edge[0], pending_edge[1], None, "entry", pending_edge[2], rep)
        if dangling is not None:
            out.append(dangling)
            out_x.append(None)
    if pending_shift is not None and pending_shift[0] != "__consumed__":
        out.append(misalignment_as_patch(elements[pending_shift[0]], pending_shift[0], rep))
        out_x.append(None)

    lat_name = name or meta.get("lattice") or getattr(line, "name", None) or "xtrack_line"
    lat = _lattice_with_shared_definitions(lat_name, out, ref, [rows.get(n) or {} for n in names],
                                           line_name=meta.get("use") or lat_name)
    lat.meta["source_format"] = "xtrack"
    xname_to_ir = _xname_map(lat, out_x, meta.get("use") or lat_name)
    _read_knobs(line, lat, xname_to_ir, rep, warn)
    if extra_lines:
        _add_extra_lines(lat, extra_lines, xname_to_ir, warn)
    if root_components and extra_lines:
        _restructure_root(lat, root_components, xname_to_ir, warn)
    if meta:
        lat.meta["xtrack"] = {k: v for k, v in meta.items() if k != "elements"}
    lat.warnings.extend(warn)

    # -- second pass: momentum steps become energy jumps, then the writer's energy mode ---
    mode = energy_mode or meta.get("energy_mode") or "local"
    check_mode(mode)
    mass0 = ref.species.mass_eV
    for e in lat.elements.values():
        if e.kind == "ReferenceChange" and e.energy_eV is None and e.dE_ref_eV is None:
            p0c = (e.native.get("xtrack") or {}).get("p0c")
            if p0c:
                e.energy_eV = math.sqrt(float(p0c) ** 2 + mass0 * mass0) - mass0
    if mode == "delta":
        undo_phase_slip(lat, rep, resolve_p0c_steps=True)      # resolves the steps on the way
    else:
        for p in propagate(lat):
            e = p.element
            if e.kind == "ReferenceChange" and e.dE_ref_eV is None and e.energy_eV is None and p.ref_in is not None:
                dp = (e.native.get("xtrack") or {}).get("Delta_p0c")
                if dp:
                    pc = p.ref_in.pc_eV + float(dp)
                    mass = p.ref_in.species.mass_eV
                    e.dE_ref_eV = math.sqrt(pc * pc + mass * mass) - mass - p.ref_in.kinetic_energy_eV
    placed = propagate(lat)
    start_brho = ref.brho_signed
    probes = probe_momentum_ratio(placed, ref)
    by_id = {id(e): n for e, n in scaled}
    done: set[int] = set()
    for p, probe in zip(placed, probes, strict=True):
        e = p.element
        if id(e) in done:
            continue
        done.add(id(e))
        brho_local = (p.ref_in or ref).brho_signed
        norm = by_id.get(id(e))
        if norm:
            _apply_rigidity(e, norm, rigidity_for(mode, brho_local, start_brho, probe))
        r = mode_ratio(mode, brho_local, start_brho, probe)
        if abs(r - 1.0) <= 1e-15:
            continue
        if e.kind == "Taylor":
            e.matrix, e.offset = scale_taylor(e.matrix, e.offset, r, inverse=True)
        elif e.kind == "Kicker":
            e.hkick, e.vkick = e.hkick / r, e.vkick / r
        elif e.kind == "Bend" and e.length:
            nat = e.native.get("xtrack") or {}
            k0 = nat.get("k0")
            g = e.bend.g_ref(e.length)
            if k0 is not None and abs(k0 - g * r) <= 1e-9 * max(1.0, abs(g * r)):
                del nat["k0"]              # the energy mode's doing, not a field/geometry split
                rep.entries = [x for x in rep.entries if not (x.element == e.name and x.code == "BEND_K0_NE_H")]
    for w in warn:
        if w not in lat.warnings:                          # pragma: no cover - defensive
            lat.warnings.append(w)
    _restore_exact_rf(lat, rows)
    return lat


def _restore_exact_rf(lat: Lattice, rows: dict) -> None:
    """The writer's exact RF numbers back onto the cavities, once the phase slips are undone:
    φ → φ + π/2 − 2π·slip → back is not bit-exact, and a field map's reference gain is its
    integral while the cavity's is V_c·cos φs (equal to 1e-12) — every downstream slip depends on
    both, so a written line is a fixed point only with the exact values restored."""
    by_ir = {r["name"]: r for r in rows.values()
             if r.get("name") and r.get("role", "main") == "main" and r.get("phase_rad") is not None}
    for name, el in lat.elements.items():
        row = by_ir.get(name)
        if row is None or el.kind != "RFCavity" or getattr(el, "rf", None) is None:
            continue
        exact = float(row["phase_rad"])
        if abs(exact - float(el.rf.phase_rad)) <= 1e-9:
            el.rf.phase_rad = exact
        if row.get("dE_ref_eV") is not None:
            gain = float(el.rf.voltage_V or 0.0) * math.cos(float(el.rf.phase_rad))
            if abs(float(row["dE_ref_eV"]) - gain) <= 1e-9 * max(1.0, abs(gain)):
                el.rf.dE_ref_eV = float(row["dE_ref_eV"])


def _apply_rigidity(el: Element, norm: dict, brho: float) -> None:
    if "ksol" in norm:
        el.solenoid.Bsol_T = norm["ksol"] * brho
    m = getattr(el, "multipole", None)
    if m is None:
        return
    for order, k in norm.get("kn", {}).items():
        m.Bn[order] = k * brho
    for order, k in norm.get("ks", {}).items():
        m.Bs[order] = k * brho
    for order, k in norm.get("knl", {}).items():
        m.BnL[order] = k * brho
    for order, k in norm.get("ksl", {}).items():
        m.BsL[order] = k * brho


#: IR expression path <-> xtrack attribute (the IR text is the *normalized* quantity, as in MAD-X)
_KNOB_ATTRS: dict[str, tuple[str, float]] = {
    "multipole.Bn[1]": ("k1", 1.0), "multipole.Bs[1]": ("k1s", 1.0), "multipole.Bn[2]": ("k2", 1.0),
    "multipole.Bs[2]": ("k2s", 1.0), "multipole.Bn[3]": ("k3", 1.0), "multipole.Bs[3]": ("k3s", 1.0),
    "length": ("length", 1.0), "bend.angle": ("angle", 1.0), "bend.e1": ("edge_entry_angle", 1.0),
    "bend.e2": ("edge_exit_angle", 1.0), "solenoid.Bsol_T": ("ks", 1.0), "rf.voltage_V": ("voltage", 1.0),
    "rf.frequency_Hz": ("frequency", 1.0), "multipole.tilt[1]": ("rot_s_rad", 1.0),
    "hkick": ("knl[0]", -1.0), "vkick": ("ksl[0]", 1.0),
}
_PATH_FOR_ATTR: dict[str, tuple[str, float]] = {}
for _path, (_attr, _sign) in _KNOB_ATTRS.items():
    _PATH_FOR_ATTR.setdefault(_attr, (_path, _sign))
for _n in range(0, 12):
    _KNOB_ATTRS.setdefault(f"multipole.BnL[{_n}]", (f"knl[{_n}]", 1.0))
    _KNOB_ATTRS.setdefault(f"multipole.BsL[{_n}]", (f"ksl[{_n}]", 1.0))
    _PATH_FOR_ATTR.setdefault(f"knl[{_n}]", (f"multipole.BnL[{_n}]", 1.0))
    _PATH_FOR_ATTR.setdefault(f"ksl[{_n}]", (f"multipole.BsL[{_n}]", 1.0))
_XDEPS_VAR = re.compile(r"vars\['([^']+)'\]")
_ELEMENT_REF = re.compile(r"element_refs\['([^']+)'\]\.([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?")
_KNOB_TOL = 1e-9


def _negate(text: str) -> str:
    """``-(text)`` without piling up unary minuses: ``-(x)`` / ``(-x)`` / ``-x`` come back as ``x``."""
    t = text.strip()

    def _wrapped(inner: str) -> bool:          # the whole of *inner* sits inside one pair of parentheses
        if not (inner.startswith("(") and inner.endswith(")")):
            return False
        depth = 0
        for i, ch in enumerate(inner):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and i < len(inner) - 1:
                return False
        return True

    name = r"[A-Za-z_][\w.]*(\[[^\]]*\])*"
    if t.startswith("-"):
        rest = t[1:].strip()
        if _wrapped(rest):
            return rest[1:-1].strip()
        if re.fullmatch(name, rest):
            return rest
    if _wrapped(t) and t[1:-1].strip().startswith("-"):
        inner = t[1:-1].strip()[1:].strip()
        if _wrapped(inner):
            return inner[1:-1].strip()
        if re.fullmatch(name, inner):
            return inner
    return f"-({t})"


def _to_ir_text(xdeps_text: str) -> str:
    """``((-vars['kq']) * 1.1)`` → ``((-kq) * 1.1)``; ``**`` → ``^`` (the IR's infix dialect)."""
    return _XDEPS_VAR.sub(r"\1", xdeps_text).replace("**", "^")


def _ref_value(line, xname: str, attr: str, index: int | None):
    obj = line[xname]
    v = getattr(obj, attr)
    if index is not None:
        v = v[index]
    return float(v)


def _read_knobs(line, lat: Lattice, xname_to_ir: dict[str, str], rep: FidelityReport, warn: list[str]) -> None:
    """xtrack's deferred expressions (``line.vars`` and ``element_refs[...]``) → IR variables and
    per-attribute :class:`~lattix.ir.expr.Expression`s (the MAD-X ``:=`` round trip)."""
    from lattix.ir.expr import Expression, evaluate
    from lattix.ir.lattice import Variable

    try:
        vars_obj = line.vars
        names = [n for n in vars_obj.keys() if n not in ("t_turn_s", "__vary_default") and not n.startswith("__")]
    except Exception:                                    # noqa: BLE001 - a line without var management
        return
    if not names:
        return
    for n in names:
        ref = vars_obj[n]
        expr = ref._expr
        value = float(ref._value)
        expression = None
        if expr is not None:
            text = _to_ir_text(str(expr))
            if "element_refs[" in text:
                warn.append(f"variable {n!r} depends on an element attribute ({text}); the value alone is kept")
            else:
                expression = Expression(text=text, deferred=True, dialect="infix")
        lat.variables[n] = Variable(value=value, expression=expression)
    values = {k: v.value for k, v in lat.variables.items()}
    try:
        pairs = line.to_dict().get("_var_manager") or []
    except Exception:                                    # noqa: BLE001
        pairs = []
    n_attr = 0
    for target, text in pairs:
        m = _ELEMENT_REF.match(str(target))
        if not m:
            continue
        xname, attr, idx = m.group(1), m.group(2), m.group(3)
        key = f"{attr}[{idx}]" if idx is not None else attr
        ir_name = xname_to_ir.get(xname)
        if ir_name is None or key not in _PATH_FOR_ATTR:
            continue
        path, sign = _PATH_FOR_ATTR[key]
        ir_text = _to_ir_text(str(text))
        if "element_refs[" in ir_text:
            continue
        if sign < 0:
            ir_text = _negate(ir_text)
        try:
            got = evaluate(ir_text, values)
        except Exception:                                # noqa: BLE001 - functions the IR does not know
            continue
        try:
            written = sign * _ref_value(line, xname, attr, int(idx) if idx is not None else None)
        except Exception:                                # noqa: BLE001
            continue
        if abs(got - written) > _KNOB_TOL * max(1.0, abs(written)):
            continue
        el = lat.elements[ir_name]
        el.expressions[path] = Expression(text=ir_text, deferred=True, dialect="infix")
        el.native.setdefault("xtrack", {})[f"{key}_expr"] = str(text)
        n_attr += 1
    if n_attr or lat.variables:
        rep.exact(None, None, code="KNOBS_READ",
                  message=f"{len(lat.variables)} variable(s) and {n_attr} deferred attribute expression(s) read")


def _attach_knobs(line, lattice: Lattice, b, rep: FidelityReport) -> None:
    """The IR's variables and deferred expressions as xtrack knobs: ``line.vars`` for the
    variables (dependent ones as xdeps expressions) and ``element_refs`` for every attribute whose
    IR expression still gives the number that was written (the MAD-X writer's rule)."""
    from lattix.ir.expr import ExpressionError, evaluate

    variables = lattice.variables
    has_expr = any(e.expressions for e in lattice.elements.values())
    if not variables and not has_expr:
        return
    values = {k: v.value for k, v in variables.items()}
    for n, var in variables.items():
        line.vars[n] = float(var.value)
    n_expr = 0
    for n, var in sorted(variables.items()):        # name order: a fixed point through a read-back
        if var.expression is None or var.expression.dialect != "infix":
            continue
        try:
            line.vars[n] = line._xdeps_eval.eval(var.expression.text)
            n_expr += 1
        except Exception as exc:                        # noqa: BLE001 - a function xdeps lacks
            rep.equivalent("KNOB_EXPRESSION_DROPPED",
                           f"variable {n!r}: expression {var.expression.text!r} is not an xdeps expression "
                           f"({type(exc).__name__}); its value is kept", element=None, kind=None)
    by_ir = {row["name"]: nm for nm, row in b.rows.items() if row.get("role", "main") == "main" and row.get("name")}
    # placement order (the order of ``line.element_names``), so that the expression table of a
    # written line is a fixed point through a read-back whatever the IR's definition order was
    order = {nm: i for i, nm in enumerate(line.element_names)}
    n_attr = 0
    for ir_name, el in sorted(lattice.elements.items(), key=lambda kv: order.get(by_ir.get(kv[0], ""), 1 << 30)):
        if not el.expressions or ir_name not in by_ir:
            continue
        xname = by_ir[ir_name]
        for path, expr in el.expressions.items():
            if path not in _KNOB_ATTRS or expr.dialect != "infix":
                continue
            attr, sign = _KNOB_ATTRS[path]
            base, idx = (attr.split("[")[0], int(attr[:-1].split("[")[1])) if "[" in attr else (attr, None)
            try:
                written = _ref_value(line, xname, base, idx)
                got = evaluate(expr.text, values)
            except (ExpressionError, Exception):         # noqa: BLE001
                continue
            if abs(sign * got - written) > _KNOB_TOL * max(1.0, abs(written)):
                continue                                  # e.g. rescaled by the energy mode: number stays
            try:
                xexpr = line._xdeps_eval.eval(_negate(expr.text) if sign < 0 else expr.text)
                if idx is not None:
                    getattr(line.element_refs[xname], base)[idx] = xexpr
                else:
                    setattr(line.element_refs[xname], base, xexpr)
                n_attr += 1
            except Exception as exc:                    # noqa: BLE001
                rep.equivalent("KNOB_EXPRESSION_DROPPED",
                               f"{ir_name}.{path}: {expr.text!r} is not an xdeps expression ({type(exc).__name__}); "
                               "the number is kept", element=ir_name, kind=el.kind)
    if variables or n_attr:
        rep.exact(None, None, code="KNOBS_WRITTEN",
                  message=f"{len(variables)} variable(s) ({n_expr} with expressions) and {n_attr} deferred "
                          "attribute expression(s) written as xtrack knobs")


def _xname_map(lat: Lattice, out_x: list[str | None], root: str) -> dict[str, str]:
    """xtrack element name -> registered IR element name, by position in the root line (the
    ``i``-th converted element is the ``i``-th item of the line, shared definitions included)."""
    items = lat.lines[root].items if root in lat.lines else []
    out: dict[str, str] = {}
    for xname, item in zip(out_x, items, strict=False):
        if xname is not None and xname not in out:
            out[xname] = item.ref
    return out


def _restructure_root(lat: Lattice, components: list[str], xname_to_ir: dict[str, str],
                      warn: list[str]) -> None:
    """The root line of an Environment as its composer wrote it (sub-lines and elements), provided
    that expansion reproduces the flat element order — so nested sources survive a read-back."""
    root = lat.lines.get(lat.use)
    if root is None:
        return
    items: list[LineItem] = []
    for c in components:
        if c in lat.lines and c != lat.use:
            items.append(LineItem(ref=c))
        elif c in xname_to_ir:
            items.append(LineItem(ref=xname_to_ir[c]))
        else:
            warn.append(f"root line: component {c!r} is neither an element nor a line; kept flat")
            return

    def expand(its, depth=0) -> list[str] | None:
        if depth > 50:
            return None
        out: list[str] = []
        for it in its:
            if it.ref in lat.lines:
                sub = expand(lat.lines[it.ref].items, depth + 1)
                if sub is None:
                    return None
                out += sub
            else:
                out.append(it.ref)
        return out

    if expand(items) == [it.ref for it in root.items]:
        root.items = items


def _add_extra_lines(lat: Lattice, extra_lines: dict[str, list[str]], xname_to_ir: dict[str, str],
                     warn: list[str]) -> None:
    """The other lines of an xtrack Environment as IR lines (components are element or line names)."""
    for lname, components in extra_lines.items():
        if lname in lat.lines:
            continue
        items: list[LineItem] = []
        for c in components:
            if c in extra_lines or c in lat.lines:
                items.append(LineItem(ref=c))
            elif c in xname_to_ir:
                items.append(LineItem(ref=xname_to_ir[c]))
            else:
                warn.append(f"line {lname!r}: component {c!r} is neither an element nor a line; skipped")
        lat.lines[lname] = Line(name=lname, items=items)


def _attach_group_extras(el: Element, names: list[str], rows: dict, elements, group) -> None:
    """Fold the sibling elements ``to_line`` emitted for one IR element back into it:
    ``LimitRect``/``LimitEllipse`` apertures and a thick collimator's drift."""
    if group is None:
        return
    roles: set[str] = set()
    for other in names:
        r = rows.get(other) or {}
        if r.get("group") != group:
            continue
        role = r.get("role", "main")
        roles.add(role)
        raw = elements[other] if isinstance(elements, dict) else elements[names.index(other)]
        if role in ("aperture_entry", "aperture_exit"):
            ap = _aperture_from_limit(raw)
            if ap is None:
                continue
            if el.aperture is None:
                el.aperture = ap
            el.aperture.aperture_at = "ENTRANCE" if role == "aperture_entry" else "EXIT"
        elif role == "extra" and type(raw).__name__ in ("Drift", "DriftExact"):
            el.length = float(raw.length)
    if el.aperture is not None and {"aperture_entry", "aperture_exit"} <= roles:
        el.aperture.aperture_at = "BOTH_ENDS"


def _aperture_from_limit(raw) -> ApertureP | None:
    cname = type(raw).__name__
    if cname == "LimitRect":
        return ApertureP(shape="RECTANGULAR",
                         x_limits=(float(raw.min_x), float(raw.max_x)),
                         y_limits=(float(raw.min_y), float(raw.max_y)))
    if cname == "LimitEllipse":
        a, b = float(raw.a), float(raw.b)
        return ApertureP(shape="ELLIPTICAL", x_limits=(-a, a), y_limits=(-b, b))
    return None


def _patch_group(line, names: list[str], rows: dict, group: int, elements, row: dict) -> Patch:
    """Re-assemble the frame elements lattix emitted for one IR ``Patch``."""
    p = Patch(name=row.get("name", f"patch_{group}"))
    for other in names:
        r = rows.get(other) or {}
        if r.get("group") != group or r.get("role") != "patch":
            continue
        raw = elements[other] if isinstance(elements, dict) else elements[names.index(other)]
        cname = type(raw).__name__
        if cname == "XYShift":
            p.x_offset, p.y_offset = float(raw.dx), float(raw.dy)
        elif cname == "SRotation":
            p.tilt = math.radians(float(raw.angle))
        elif cname == "XRotation":
            p.x_rot = math.radians(float(raw.angle))
        elif cname == "YRotation":
            p.y_rot = math.radians(float(raw.angle))
        elif cname in ("Drift", "DriftExact"):
            p.z_offset = float(raw.length)
    p.provenance = Provenance(format="xtrack", original_name=row.get("original_name"),
                              original_type="Patch")
    return p


def _convert_element(raw, xname: str, row: dict, rep: FidelityReport,
                     warn: list[str]) -> tuple[Element | None, dict]:
    """One xtrack element → (IR element, normalized strengths to scale by Bρ later)."""
    import numpy as np

    target, weight, slice_kind = _unslice(raw)
    if slice_kind:
        sname = type(raw).__name__
        n0 = len(rep.entries)
        el, norm = _slice_to_ir(raw, target, weight, slice_kind, xname, row, rep, warn)
        del rep.entries[n0:]                   # the parent's own entry is not this element's
        rep.equivalent("SLICE_MERGED",
                       f"xtrack {sname} mapped to its parent element scaled by "
                       f"weight={weight:g}", element=xname, kind=sname, weight=weight)
        if el is not None:
            el.provenance = Provenance(format="xtrack", original_name=xname,
                                       original_type=sname)
        return el, norm

    cname = type(raw).__name__
    ir_name = row.get("name") or xname
    ir_kind = row.get("kind")
    prov = Provenance(format="xtrack", original_name=row.get("original_name") if row else xname,
                      original_type=row.get("original_type") or cname)
    norm: dict = {}
    n_entries = len(rep.entries)

    def finish(el: Element) -> tuple[Element, dict]:
        el.provenance = prov
        sh = _shift_from(raw)
        if sh is not None:
            el.shift = sh
        if len(rep.entries) == n_entries:      # exactly one ledger entry per element
            rep.exact(ir_name, el.kind)
        return el, norm

    if cname in ("Drift", "DriftExact"):
        return finish(Drift(name=ir_name, length=float(raw.length)))

    if cname == "Quadrupole":
        el = Quadrupole(name=ir_name, length=float(raw.length))
        norm["kn"] = {1: float(raw.k1)}
        if float(raw.k1s):
            norm["ks"] = {1: float(raw.k1s)}
        if float(raw.rot_s_rad):
            el.multipole.tilt[1] = float(raw.rot_s_rad)
        return finish(el)

    if cname == "Sextupole":
        el = Sextupole(name=ir_name, length=float(raw.length))
        norm["kn"] = {2: float(raw.k2)}
        if float(raw.k2s):
            norm["ks"] = {2: float(raw.k2s)}
        if float(raw.rot_s_rad):
            el.multipole.tilt[2] = float(raw.rot_s_rad)
        return finish(el)

    if cname == "Octupole":
        el = Octupole(name=ir_name, length=float(raw.length))
        norm["kn"] = {3: float(raw.k3)}
        if float(raw.k3s):
            norm["ks"] = {3: float(raw.k3s)}
        if float(raw.rot_s_rad):
            el.multipole.tilt[3] = float(raw.rot_s_rad)
        return finish(el)

    if cname == "Multipole":
        knl = [float(v) for v in np.atleast_1d(raw.knl)]
        ksl = [float(v) for v in np.atleast_1d(raw.ksl)]
        hxl = float(getattr(raw, "hxl", 0.0) or 0.0)
        raw_length = float(getattr(raw, "length", 0.0) or 0.0)
        # a Multipole only occupies space when isthick is set; otherwise `length` is lrad
        isthick = bool(getattr(raw, "isthick", False))
        length = raw_length if isthick else 0.0
        lrad = 0.0 if isthick else raw_length
        thin_dipole = (len(knl) <= 1 and len(ksl) <= 1 and hxl == 0.0)
        if ir_kind == "Kicker" or (ir_kind is None and thin_dipole):
            el = Kicker(name=ir_name, length=length,
                        hkick=-(knl[0] if knl else 0.0), vkick=(ksl[0] if ksl else 0.0))
            if lrad:
                el.native.setdefault("xtrack", {})["lrad"] = lrad
            if ir_kind is None:
                rep.equivalent("MULTIPOLE_AS_KICKER",
                               "a thin xt.Multipole with only an order-0 component is "
                               "xtrack's own encoding of a MAD-X kicker "
                               "(knl[0] = -hkick, ksl[0] = +vkick)",
                               element=ir_name, kind="Kicker")
            el.provenance = prov
            sh = _shift_from(raw)
            if sh is not None:
                el.shift = sh
            if len(rep.entries) == n_entries:
                rep.exact(ir_name, "Kicker")
            return el, {}
        el = Multipole(name=ir_name, length=length)
        if lrad:
            el.native.setdefault("xtrack", {})["lrad"] = lrad
        norm["knl"] = {i: v for i, v in enumerate(knl) if v}
        norm["ksl"] = {i: v for i, v in enumerate(ksl) if v}
        if hxl:
            el.native.setdefault("xtrack", {})["hxl"] = hxl
            rep.lossy("MULTIPOLE_HXL_KEPT_NATIVE",
                      "xt.Multipole.hxl is a geometric bend the IR Multipole cannot hold; "
                      "kept as a native passthrough and re-emitted to xtrack",
                      element=ir_name, kind="Multipole", hxl=hxl)
        if float(raw.rot_s_rad):
            el.multipole.tilt[0] = float(raw.rot_s_rad)
        return finish(el)

    if cname in ("Bend", "RBend"):
        length = float(raw.length)
        h = float(raw.h)
        e1, e2 = float(raw.edge_entry_angle), float(raw.edge_exit_angle)
        if cname == "RBend":
            # xt.RBend face angles are relative to the rectangular faces (MAD-X rbend e1/e2); the IR
            # keeps sector-referenced angles, so add the wedge each face makes with the sector
            # (θ/2, split unevenly by rbend_angle_diff).  Measured against cpymad: 5.1e-10.
            angle = h * length
            diff = float(getattr(raw, "rbend_angle_diff", 0.0) or 0.0)
            e1 += 0.5 * angle - 0.5 * diff
            e2 += 0.5 * angle + 0.5 * diff
        el = Bend(name=ir_name, length=length,
                  bend=BendP(angle=h * length,
                             e1=e1, e2=e2,
                             edge_int1=float(raw.edge_entry_fint),
                             edge_int2=float(raw.edge_exit_fint),
                             hgap=float(raw.edge_entry_hgap),
                             tilt_ref=float(raw.rot_s_rad),
                             rect=(cname == "RBend")))
        k0 = _bend_k0(raw)
        if not bool(raw.k0_from_h) and abs(k0 - h) > 1e-15 * max(1.0, abs(h)):
            el.native.setdefault("xtrack", {})["k0"] = k0
            rep.lossy("BEND_K0_NE_H",
                      "the dipole field (k0) differs from the reference curvature (h); the IR "
                      "Bend holds only the geometry, k0 kept as a native passthrough",
                      element=ir_name, kind="Bend", k0=k0, h=h)
        norm["kn"] = {}
        if float(raw.k1):
            norm["kn"][1] = float(raw.k1)
        if float(raw.k2):
            norm["kn"][2] = float(raw.k2)
        knl = [float(v) for v in np.atleast_1d(raw.knl)]
        ksl = [float(v) for v in np.atleast_1d(raw.ksl)]
        if any(knl) or any(ksl):
            norm["knl"] = {i: v for i, v in enumerate(knl) if v}
            norm["ksl"] = {i: v for i, v in enumerate(ksl) if v}
            rep.equivalent("BEND_THIN_MULTIPOLES_FOLDED",
                           "the bend's thin knl/ksl kicks were folded into the IR Bend's "
                           "integrated multipole content", element=ir_name, kind="Bend")
        return finish(el)

    if cname in ("UniformSolenoid", "Solenoid"):
        el = Solenoid(name=ir_name, length=float(raw.length), solenoid=SolenoidP())
        norm["ksol"] = float(raw.ks)
        if float(getattr(raw, "ksi", 0.0) or 0.0):
            rep.lossy("THIN_SOLENOID_KSI_DROPPED",
                      "xt.Solenoid.ksi (integrated thin solenoid) has no IR form",
                      element=ir_name, kind="Solenoid", ksi=float(raw.ksi))
        return finish(el)

    if cname == "Cavity":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            lag = float(raw.lag)
        rf = RFP(frequency_Hz=float(raw.frequency) or None,
                 voltage_V=float(raw.voltage),
                 phase_rad=phase_from_xtrack(lag, float(raw.phase)))
        harmonic = float(getattr(raw, "harmonic", 0.0) or 0.0)
        if harmonic:
            rf.harmon = harmonic
        el = RFCavity(name=ir_name, length=float(getattr(raw, "length", 0.0) or 0.0), rf=rf)
        if abs(rf.voltage_V * math.cos(rf.phase_rad)) > _ACCEL_TOL_eV:
            rep.equivalent("CONST_P0",
                           "xtrack keeps p0c constant across RF: the source line's reference "
                           "momentum did not follow this cavity's gain",
                           element=ir_name, kind="RFCavity")
        return finish(el)

    if cname == "Marker":
        return finish(Marker(name=ir_name))

    if cname in ("LimitRect", "LimitEllipse"):
        el = Collimator(name=ir_name, aperture=_aperture_from_limit(raw))
        return finish(el)

    if cname in ("XYShift", "SRotation", "XRotation", "YRotation"):
        el = Patch(name=ir_name)
        if cname == "XYShift":
            el.x_offset, el.y_offset = float(raw.dx), float(raw.dy)
        elif cname == "SRotation":
            el.tilt = math.radians(float(raw.angle))
        elif cname == "XRotation":
            el.x_rot = math.radians(float(raw.angle))
        else:
            el.y_rot = math.radians(float(raw.angle))
        return finish(el)

    if cname == "ZetaShift":
        el = ReferenceChange(name=ir_name)
        el.native.setdefault("xtrack", {})["dzeta"] = float(raw.dzeta)
        rep.equivalent("ZETASHIFT_AS_REFCHANGE",
                       "xt.ZetaShift written as a reference-time change; the time offset "
                       "needs the local beta, kept as a native passthrough",
                       element=ir_name, kind="ReferenceChange", dzeta=float(raw.dzeta))
        return finish(el)

    if cname == "ReferenceEnergyIncrease":
        el = ReferenceChange(name=ir_name)
        el.native.setdefault("xtrack", {})["Delta_p0c"] = float(raw.Delta_p0c)
        rep.equivalent("REFCHANGE_AS_P0C",
                       "xt.ReferenceEnergyIncrease is a momentum step; its energy equivalent "
                       "depends on the local reference and is resolved by the walk",
                       element=ir_name, kind="ReferenceChange", Delta_p0c=float(raw.Delta_p0c))
        return finish(el)

    if cname == "FirstOrderTaylorMap":
        m1 = np.asarray(raw.m1, dtype=float).reshape(6, 6)
        m0 = np.asarray(getattr(raw, "m0", np.zeros(6)), dtype=float).reshape(6)
        el = Taylor(name=ir_name, length=float(getattr(raw, "length", 0.0) or 0.0),
                    matrix=m1.tolist(), offset=m0.tolist())
        rep.equivalent("TAYLOR_BASIS_XTRACK",
                       "the map is stored in xtrack's (x, px, y, py, zeta, delta) basis",
                       element=ir_name, kind="Taylor")
        el.provenance = prov
        return el, {}

    # -- the rest of the zoo (Phase 5.1) -----------------------------------
    from lattix.formats.xtrack.extra_elements import convert_extra

    extra = convert_extra(cname, raw, ir_name, rep, norm, warn)
    if extra is not None:
        return finish(extra)

    # -- anything else -----------------------------------------------------
    el = Marker(name=ir_name)
    el.provenance = prov
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            payload = raw.to_dict()
        el.native["xtrack"] = {"element": _plain(payload)}
    except Exception:                                     # noqa: BLE001 - best effort
        el.native["xtrack"] = {"class": cname}
    rep.dropped("UNSUPPORTED_XTRACK_ELEMENT",
                f"xtrack element class {cname!r} has no IR kind; written as a marker with a "
                "verbatim native passthrough", element=ir_name, kind="Marker",
                xtrack_class=cname)
    return el, {}


#: parent attribute holding the per-metre strength of each thick class, and the
#: multipole order it integrates into when a *thin* slice is taken.
_THIN_ORDERS = {"Quadrupole": (("k1", 1), ("k1s", 1)),
                "Sextupole": (("k2", 2), ("k2s", 2)),
                "Octupole": (("k3", 3), ("k3s", 3)),
                "Bend": (("k0", 0), ("k1", 1), ("k2", 2)),
                "RBend": (("k0", 0), ("k1", 1), ("k2", 2))}


def _bend_k0(raw) -> float:
    """``Bend.k0`` reads back as the *string* ``'from_h'`` while ``k0_from_h`` is set
    (measured on 0.103.5 and 0.112.0), so never ``float()`` it blind."""
    return float(raw.h) if bool(raw.k0_from_h) else float(raw.k0)


def _slice_to_ir(raw, parent, weight: float, slice_kind: str, xname: str, row: dict,
                 rep: FidelityReport, warn: list[str]) -> tuple[Element | None, dict]:
    """``DriftSlice*`` / ``ThickSlice*`` / ``ThinSlice*`` → an IR element (EQUIVALENT)."""
    ir_name = row.get("name") or xname
    if slice_kind == "drift":
        base = getattr(raw, "_parent", None)
        return Drift(name=ir_name, length=float(getattr(base, "length", 0.0)) * weight), {}

    pname = type(parent).__name__
    if slice_kind == "thick":
        el, norm = _convert_element(parent, xname, row, rep, warn)
        if el is None:                                  # pragma: no cover - defensive
            return None, {}
        el.name = ir_name
        el.length = el.length * weight
        if isinstance(el, Bend):
            el.bend.angle = el.bend.angle * weight
            el.bend.e1 = el.bend.e2 = 0.0               # edges are separate entry/exit slices
        return el, norm

    # thin slice: integrated kicks = per-metre strength × parent length × weight
    length = float(getattr(parent, "length", 0.0) or 0.0)
    scale = length * weight
    norm: dict = {"knl": {}, "ksl": {}}
    for attr, order in _THIN_ORDERS.get(pname, ()):
        value = _bend_k0(parent) if attr == "k0" else float(getattr(parent, attr, 0.0) or 0.0)
        if not value:
            continue
        key = "ksl" if attr.endswith("s") else "knl"
        norm[key][order] = norm[key].get(order, 0.0) + value * scale
    el = Multipole(name=ir_name, length=0.0, multipole=MagneticMultipoleP())
    hxl = float(getattr(parent, "h", 0.0) or 0.0) * scale
    if hxl:
        el.native.setdefault("xtrack", {})["hxl"] = hxl
    if not norm["knl"] and not norm["ksl"] and not hxl:
        return Marker(name=ir_name), {}
    return el, norm


def _shift_from(raw) -> BodyShiftP | None:
    if not _has_shift_fields(type(raw)):
        return None
    if type(raw).__name__ in ("Translation", "Rotation"):     # frame patches, not misaligned bodies
        return None

    def g(attr: str) -> float:
        try:
            return float(getattr(raw, attr))
        except (AttributeError, TypeError, ValueError):
            return 0.0

    sh = BodyShiftP(x_offset=g("shift_x"), y_offset=g("shift_y"), z_offset=g("shift_s"),
                    tilt=g("rot_s_rad_no_frame"), x_rot=g("rot_x_rad"), y_rot=g("rot_y_rad"))
    return None if sh.is_zero() else sh


def _plain(obj):
    """JSON-friendly copy of an xtrack ``to_dict`` payload (numpy → list/float)."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
