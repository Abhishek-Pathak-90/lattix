"""IR ⇄ HELIX (``linac_gen``) adapter — Phase 1 task 1.6.

HELIX is GPL-3 and private; it is **never** a dependency of lattix.  The
package is imported at call time from ``HELIX_ROOT`` through
:func:`lattix.oracles.helix._import_helix`, so importing this module costs
nothing when HELIX is absent.  This is an *adapter*, not a file format:
it is deliberately absent from :data:`lattix.formats.base.FORMATS`.

Units: HELIX works in mm / deg / MeV / MHz / T / T·m; the IR is SI + eV.

Phase convention (PLAN §4.1, ``lattix.ir.rf``): the IR phase is the
SYNCHRONOUS phase of the reference particle (0 = crest, gain =
``V_eff·cos φ``, species-independent).  HELIX's thin ``RFGap.advance_ref``
uses the SIGNED charge (``dW = q·V·T·cos φ``), so an H⁻ deck's raw deck
phase maps to ``φ + π``; :func:`lattix.ir.rf.phase_from_tracewin_deg`
does that.

How HELIX decides "sync-phase mode" (measured, not recalled):

* ``linac_gen/io/tracewin_parser.py:1041`` — a ``SET_SYNC_PHASE`` card sets
  the one-shot parser flag ``pending_sync_phase``.
* ``linac_gen/io/tracewin_parser.py:310`` — the next ``FIELD_MAP`` consumes
  it and is built with ``p_flag = 1``; ``linac_gen/elements/field_map.py:292``
  (``_phi_sync_rad``) and ``field_map.py:921`` then run the iterative
  ``_calibrate_sync_phase`` scan instead of the relative-phase branch.
* ``linac_gen/io/tracewin_parser.py:975`` — an ``NCELLS`` card consumes it
  into the constructor flag ``NCells(sync_phase=...)``.
* ``linac_gen/io/tracewin_parser.py:701`` — a thin ``GAP`` does **not**:
  HELIX warns that ``SET_SYNC_PHASE`` binds only to the next FIELD_MAP and
  keeps the relative phase (``tracewin_parser.py:960`` additionally
  downgrades ``GAP`` cards whose own ``p_flag != 0``).  ``RFGap`` therefore
  has no sync-phase mode at all, and this adapter converts every ``RFGap``
  with ``sync=False`` (recording ``SYNC_MODE_ASSUMED``) — the only mode in
  which HELIX's own ``advance_ref`` reproduces the IR reference gain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lattix.fidelity import FidelityReport
from lattix.ir.elements import (
    RFP,
    ApertureP,
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
    Provenance,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    RFQCell,
    Solenoid,
    SolenoidP,
    Superposition,
    Taylor,
)
from lattix.ir.elements import Bend as IRBend
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle
from lattix.ir.reference import species as ir_species
from lattix.ir.rf import phase_from_tracewin_deg, tracewin_phase_deg
from lattix.ir.walk import energy_gain_eV as _ir_energy_gain
from lattix.ir.walk import propagate
from lattix.oracles.helix import _import_helix

FORMAT = "helix"
MM = 1e-3
MEV = 1e6
MHZ = 1e6
DEFAULT_FREQUENCY_HZ = 352.21e6      # tracewin_parser._DEFAULT_FREQ_MHZ
DEFAULT_FRINGE_K2 = 2.80             # tracewin_syntax.SCHEMA["EDGE"] default

# species name (lattix) -> attribute of linac_gen.core.particle
_SPECIES_ATTR = {"proton": "PROTON", "h-": "H_MINUS", "deuteron": "DEUTERON"}
_FIELD_ERROR_ATTRS = ("gradient_rel", "field_rel", "voltage_rel", "phase_offset",
                      "frequency_offset")

__all__ = ["FORMAT", "RULES", "from_helix", "to_helix", "read_helix_deck"]


# ---------------------------------------------------------------------------
# HELIX import (lazy, cached)
# ---------------------------------------------------------------------------
_MODS: Any = None


def _helix() -> Any:
    """Import ``linac_gen`` once and cache the classes this adapter needs."""
    global _MODS
    if _MODS is not None:
        return _MODS
    root = _import_helix()
    from types import SimpleNamespace

    from linac_gen.core.lattice import Lattice as HLattice
    from linac_gen.core.particle import DEUTERON, H_MINUS, PROTON
    from linac_gen.core.reference import ReferenceParticle as HRef
    from linac_gen.elements.aperture import Aperture as HAperture
    from linac_gen.elements.base import FieldMapElement, ThinKickElement
    from linac_gen.elements.dipole import Dipole as HDipole
    from linac_gen.elements.drift import Drift as HDrift
    from linac_gen.elements.edge import Edge as HEdge
    from linac_gen.elements.field_map import FieldMap as HFieldMap
    from linac_gen.elements.field_map_3d import FieldMap3D as HFieldMap3D
    from linac_gen.elements.field_map_factory import make_field_map_element
    from linac_gen.elements.foil import Foil as HFoil
    from linac_gen.elements.lattice_commands import (
        COMMAND_CLASSES,
        LatticeCommand,
        SetBeamE0P0,
        SetBeamEnergy,
        SetSyncPhase,
    )
    from linac_gen.elements.lattice_commands import Freq as HFreq
    from linac_gen.elements.marker import Marker as HMarker
    from linac_gen.elements.matrix_element import MatrixElement as HMatrix
    from linac_gen.elements.mixins import Misalignment
    from linac_gen.elements.multipole import Multipole as HMultipole
    from linac_gen.elements.ncells import NCells as HNCells
    from linac_gen.elements.ncells import TTFSet, TTFTable
    from linac_gen.elements.quadrupole import Quadrupole as HQuad
    from linac_gen.elements.rf_gap import RFGap as HRFGap
    from linac_gen.elements.sc_grid import ScGridDirective
    from linac_gen.elements.solenoid import Solenoid as HSolenoid
    from linac_gen.elements.space_charge_comp import SpaceChargeComp
    from linac_gen.elements.steerer import Steerer as HSteerer
    from linac_gen.elements.superposed_field_map import SuperposedFieldMap
    from linac_gen.elements.thin_lens import ThinLens as HThinLens
    from linac_gen.io.tracewin_syntax import SCHEMA, parse_positionals

    try:
        from linac_gen.elements.rfq_cell import RfqCell as HRfqCell
    except Exception:                                        # pragma: no cover
        HRfqCell = None
    try:
        from linac_gen.elements.vane_rfq import VaneRFQ as HVaneRFQ
    except Exception:                                        # pragma: no cover
        HVaneRFQ = None

    _MODS = SimpleNamespace(
        root=root, Lattice=HLattice, Ref=HRef,
        PROTON=PROTON, H_MINUS=H_MINUS, DEUTERON=DEUTERON,
        FieldMapElement=FieldMapElement, ThinKickElement=ThinKickElement,
        Misalignment=Misalignment,
        Aperture=HAperture, Dipole=HDipole, Drift=HDrift, Edge=HEdge,
        FieldMap=HFieldMap, FieldMap3D=HFieldMap3D, Foil=HFoil, Marker=HMarker,
        MatrixElement=HMatrix, Multipole=HMultipole, NCells=HNCells,
        Quadrupole=HQuad, RFGap=HRFGap, Solenoid=HSolenoid, Steerer=HSteerer,
        SuperposedFieldMap=SuperposedFieldMap, ThinLens=HThinLens,
        SpaceChargeComp=SpaceChargeComp, ScGridDirective=ScGridDirective,
        RfqCell=HRfqCell, VaneRFQ=HVaneRFQ,
        LatticeCommand=LatticeCommand, COMMAND_CLASSES=COMMAND_CLASSES,
        Freq=HFreq, SetSyncPhase=SetSyncPhase, SetBeamEnergy=SetBeamEnergy,
        SetBeamE0P0=SetBeamE0P0,
        TTFSet=TTFSet, TTFTable=TTFTable,
        make_field_map_element=make_field_map_element,
        SCHEMA=SCHEMA, parse_positionals=parse_positionals,
    )
    return _MODS


# ---------------------------------------------------------------------------
# small unit helpers (native-preserving so a HELIX → IR → HELIX trip is exact)
# ---------------------------------------------------------------------------
def _to_mm(value_m: float, native_mm: float | None = None) -> float:
    """metres → mm, returning the untouched source value when it still matches."""
    if native_mm is not None and abs(value_m - native_mm * MM) <= 1e-12 * max(1.0, abs(value_m)):
        return float(native_mm)
    return value_m / MM


def _to_deg(value_rad: float, native_deg: float | None = None) -> float:
    if native_deg is not None and abs(value_rad - math.radians(native_deg)) <= 1e-12 * max(
            1.0, abs(value_rad)):
        return float(native_deg)
    return math.degrees(value_rad)


def _aperture_p(radius_mm: float | None, radius_y_mm: float | None = None) -> ApertureP | None:
    """HELIX ``aperture`` (radius, mm) → PALS ApertureP (elliptical)."""
    if not radius_mm or radius_mm <= 0:
        return None
    rx = float(radius_mm) * MM
    ry = float(radius_y_mm) * MM if radius_y_mm else rx
    return ApertureP(shape="ELLIPTICAL", x_limits=(-rx, rx), y_limits=(-ry, ry))


def _aperture_mm(ap: ApertureP | None, native: dict | None = None) -> tuple[float, float | None]:
    if native and "aperture" in native:
        return float(native["aperture"] or 0.0), native.get("aperture_y")
    if ap is None or ap.half_x is None:
        return 0.0, None
    hx = ap.half_x / MM
    hy = None if ap.half_y is None or abs(ap.half_y - ap.half_x) < 1e-15 else ap.half_y / MM
    return hx, hy


def _shift_p(el: Any, mods: Any) -> BodyShiftP | None:
    """HELIX Misalignment mixin (mm / deg) → PALS BodyShiftP (m / rad)."""
    if not isinstance(el, mods.Misalignment):
        return None
    s = BodyShiftP(
        x_offset=float(el.dx) * MM, y_offset=float(el.dy) * MM, z_offset=float(el.dz) * MM,
        x_rot=math.radians(float(el.pitch_deg)), y_rot=math.radians(float(el.yaw_deg)),
        tilt=math.radians(float(el.tilt_deg)),
    )
    return None if s.is_zero() else s


def _shift_kwargs(shift: BodyShiftP | None) -> dict:
    if shift is None:
        return {}
    return {"dx": shift.x_offset / MM, "dy": shift.y_offset / MM, "dz": shift.z_offset / MM,
            "tilt_deg": math.degrees(shift.tilt), "pitch_deg": math.degrees(shift.x_rot),
            "yaw_deg": math.degrees(shift.y_rot)}


def _field_errors(el: Any) -> dict:
    return {a: float(getattr(el, a)) for a in _FIELD_ERROR_ATTRS
            if float(getattr(el, a, 0.0) or 0.0) != 0.0}


def _jsonable(value: Any) -> Any:
    """Best-effort JSON-safe copy so ``Lattice.to_dict()`` never explodes."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {"__class__": type(value).__name__,
                **{k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}}
    return repr(value)


# ---------------------------------------------------------------------------
# from_helix
# ---------------------------------------------------------------------------
@dataclass
class _FromCtx:
    mods: Any
    rep: FidelityReport
    charge: int
    ir_ref: ReferenceParticle
    pending_sync: bool = False
    superpositions: list = field(default_factory=list)   # (Superposition, [(z0_m, Element)])


def _helix_species(name: str, mods: Any):
    key = str(name).strip().lower()
    key = {"p": "proton", "hminus": "h-", "h_minus": "h-", "d": "deuteron"}.get(key, key)
    attr = _SPECIES_ATTR.get(key)
    if attr is None:
        raise ValueError(f"HELIX has no species {name!r} (proton, h-, deuteron)")
    return getattr(mods, attr), key


def _first_frequency_hz(helix_lattice, mods: Any) -> float | None:
    for e in helix_lattice.elements:
        if isinstance(e, mods.Freq):
            return float(e.frequency_mhz) * MHZ
        f = getattr(e, "frequency", None)
        if f:
            return float(f) * MHZ
    return None


def _advance_helix_ref(el: Any, href: Any, mods: Any) -> None:
    """Advance the HELIX reference exactly as ``lattix.oracles.helix.HelixOracle.run``
    (itself a mirror of ``tracking.matrix_tracking.compute_transfer_matrix``)."""
    if isinstance(el, mods.Freq):
        if el.frequency_mhz:
            href.frequency = float(el.frequency_mhz)
        return
    if isinstance(el, mods.SetBeamEnergy):
        href.w_kin = float(el.energy_MeV)
        return
    if isinstance(el, mods.FieldMapElement):
        el.reset_run_state()
        el.advance_ref(href)
        return
    href.s += el.length
    if el.length > 0 and href.wavelength > 0:
        href.phi_s += 360.0 * el.length / (href.beta * href.wavelength)
    if isinstance(el, mods.ThinKickElement):
        el.advance_ref(href)


def _bend_group(elements: list, i: int, mods: Any) -> tuple[Any, Any, Any, int]:
    """Cluster ``EDGE? BEND EDGE?`` starting at *i*; returns (e_in, dipole, e_out, n)."""
    n = len(elements)
    e_in = dipole = e_out = None
    j = i
    if isinstance(elements[j], mods.Edge) and j + 1 < n and isinstance(elements[j + 1], mods.Dipole):
        e_in = elements[j]
        j += 1
    dipole = elements[j]
    j += 1
    # a trailing EDGE belongs to this bend unless it is the entrance edge of the next one
    if j < n and isinstance(elements[j], mods.Edge) and not (
            j + 1 < n and isinstance(elements[j + 1], mods.Dipole)):
        e_out = elements[j]
        j += 1
    return e_in, dipole, e_out, j - i


def _convert_bend(ctx: _FromCtx, e_in, dip, e_out) -> Element:
    mods = ctx.mods
    rep = ctx.rep
    angle_rad = math.radians(float(dip.angle))
    rho_mm = float(dip.rho)
    rho_m = abs(rho_mm) * MM
    edges = [x for x in (e_in, e_out) if x is not None]
    if edges and (dip.e1 or dip.e2):
        rep.lossy("BEND_EDGE_CONFLICT",
                  "HELIX Dipole carries e1/e2 AND adjacent EDGE cards; the EDGE cards win",
                  element=str(dip.name), kind="Bend",
                  dipole_e1=float(dip.e1), dipole_e2=float(dip.e2),
                  edge_in=None if e_in is None else float(e_in.pole_rotation),
                  edge_out=None if e_out is None else float(e_out.pole_rotation))
    sgn = 1.0 if angle_rad >= 0 else -1.0
    # HELIX Edge.pole_rotation β = sign(θ)·e (madx_parser.py:445-449)
    if e_in is not None:
        e1 = sgn * math.radians(float(e_in.pole_rotation))
    else:
        e1 = sgn * math.radians(float(dip.e1))
    if e_out is not None:
        e2 = sgn * math.radians(float(e_out.pole_rotation))
    else:
        e2 = sgn * math.radians(float(dip.e2))
    gap_mm = float(edges[0].gap) if edges else 0.0
    k2 = float(edges[0].k2) if edges else None
    bend = BendP(
        angle=angle_rad, e1=e1, e2=e2,
        edge_int1=float(e_in.k1) if e_in is not None else (float(e_out.k1) if e_out is not None else 0.0),
        edge_int2=float(e_out.k1) if e_out is not None else None,
        hgap=gap_mm * MM / 2.0,
        tilt_ref=math.pi / 2 if int(dip.hv) else 0.0,
        fringe_k2=k2,
    )
    mp = MagneticMultipoleP()
    if float(dip.field_index) and rho_m > 0:
        mp.Bn[1] = -float(dip.field_index) * ctx.ir_ref.brho_signed / (rho_m * rho_m)
    native = {"class": "Dipole", "angle_deg": float(dip.angle), "rho_mm": rho_mm,
              "field_index": float(dip.field_index), "hv": int(dip.hv),
              "length_mm": float(dip.length), "aperture": float(dip.aperture)}
    if edges:
        native["edges"] = [
            None if x is None else {"name": str(x.name), "pole_rotation": float(x.pole_rotation),
                                    "rho": float(x.rho), "gap": float(x.gap), "k1": float(x.k1),
                                    "k2": float(x.k2), "aperture": float(x.aperture_radius),
                                    "hv": int(x.hv)}
            for x in (e_in, e_out)]
    fe = _field_errors(dip)
    if fe:
        native["field_errors"] = fe
    el = IRBend(name=str(dip.name), length=float(dip.length) * MM, bend=bend, multipole=mp,
                aperture=_aperture_p(dip.aperture), shift=_shift_p(dip, mods),
                tracking={"n_steps": int(dip.n_steps)}, native={FORMAT: native},
                provenance=Provenance(format=FORMAT, original_name=str(dip.name),
                                      original_type="BEND"))
    if fe:
        rep.lossy("FIELD_ERROR_NOT_CONVERTED", "HELIX per-seed field error dropped",
                  element=el.name, kind="Bend", **fe)
    elif not edges and (dip.e1 or dip.e2):
        rep.equivalent("BEND_EDGES_FROM_DIPOLE",
                       "e1/e2 taken from the Dipole (no EDGE cards present)",
                       element=el.name, kind="Bend")
    else:
        rep.exact(el.name, "Bend")
    return el


def _convert_marker(ctx: _FromCtx, e: Any) -> Element:
    native = {"class": "Marker", "snapshot": bool(e.snapshot), "is_bpm": bool(e.is_bpm),
              "diag_family": e.diag_family, "x_target_mm": e.x_target_mm,
              "y_target_mm": e.y_target_mm, "accuracy_mm": float(e.accuracy_mm),
              "origin_keyword": e.origin_keyword}
    for extra in ("lattice_card_args", "origin_params"):
        if hasattr(e, extra):
            native[extra] = _jsonable(getattr(e, extra))
    family = None
    if e.origin_keyword:
        family = str(e.origin_keyword).upper()
    elif e.is_bpm:
        family = "BPM"
    elif e.snapshot:
        family = "DIAG_PHASE"
    prov = Provenance(format=FORMAT, original_name=str(e.name), original_type="MARKER")
    if family is None:
        ctx.rep.exact(str(e.name), "Marker")
        return Marker(name=str(e.name), native={FORMAT: native}, provenance=prov)
    params = {k: native[k] for k in ("diag_family", "x_target_mm", "y_target_mm", "accuracy_mm")
              if native[k] is not None}
    ctx.rep.exact(str(e.name), "Instrument")
    return Instrument(name=str(e.name), family=family, params=params, native={FORMAT: native},
                      provenance=prov)


def _convert_command(ctx: _FromCtx, e: Any) -> Element:
    mods, rep = ctx.mods, ctx.rep
    name = str(e.name)
    kwargs = {k: _jsonable(v) for k, v in vars(e).items()
              if not k.startswith("_") and k not in ("name", "length", "aperture", "n_steps")}
    card = type(e).KEYWORD
    native = {"class": type(e).__name__, "card": card, "kwargs": kwargs}
    prov = Provenance(format=FORMAT, original_name=name, original_type=card)
    if isinstance(e, mods.Freq):
        rep.exact(name, "Freq")
        return Freq(name=name, frequency_Hz=float(e.frequency_mhz) * MHZ,
                    native={FORMAT: native}, provenance=prov)
    if isinstance(e, mods.SetBeamEnergy):
        rep.exact(name, "ReferenceChange")
        return ReferenceChange(name=name, energy_eV=float(e.energy_MeV) * MEV,
                               native={FORMAT: native}, provenance=prov)
    if isinstance(e, mods.SetBeamE0P0):
        dE = float(e.dE_MeV) * MEV if int(e.ke) else 0.0
        if int(e.kp):
            rep.lossy("REF_PHASE_SHIFT_NOT_CONVERTED",
                      "SET_BEAM_E0_P0 phase shift has no IR equivalent",
                      element=name, kind="ReferenceChange", dphi_deg=float(e.dphi_deg))
        else:
            rep.exact(name, "ReferenceChange")
        return ReferenceChange(name=name, dE_ref_eV=dE, native={FORMAT: native}, provenance=prov)
    if isinstance(e, mods.SetSyncPhase):
        ctx.pending_sync = True
        rep.exact(name, "Directive")
        return Directive(name=name, format="tracewin", card=card, args=[], role="sync_phase",
                         native={FORMAT: native}, provenance=prov)
    rep.exact(name, "Directive")
    return Directive(name=name, format="tracewin", card=card,
                     args=[str(a) for a in e.to_tracewin_args()], role="matching",
                     native={FORMAT: native}, provenance=prov)


def _convert_fieldmap(ctx: _FromCtx, e: Any, dE_eV: float) -> Element:
    mods, rep = ctx.mods, ctx.rep
    sync = int(getattr(e, "p_flag", 0)) == 1
    if sync:
        ctx.pending_sync = False
    native = {"class": type(e).__name__, "length_mm": float(e.length), "scale": float(e.scale),
              "phase_deg": float(e.phase), "frequency_mhz": float(e.frequency),
              "aperture": float(e.aperture), "n_steps": int(e.n_steps),
              "p_flag": int(e.p_flag), "geom": e.geom, "field_file": e.field_file}
    fe = _field_errors(e)
    if fe:
        native["field_errors"] = fe
    rf = RFP(frequency_Hz=float(e.frequency) * MHZ,
             phase_rad=phase_from_tracewin_deg(float(e.phase), ctx.charge, sync),
             dE_ref_eV=dE_eV, phase_is_sync=sync,
             L_active_m=float(e.length) * MM)
    el = FieldMap(name=str(e.name), length=float(e.length) * MM, rf=rf, geom=e.geom,
                  files=[e.field_file] if e.field_file else [],
                  ke=float(e.ke), kb=float(e.kb), ki=float(e.ki), ka=float(e.ka),
                  p_flag=int(e.p_flag), aperture=_aperture_p(e.aperture),
                  shift=_shift_p(e, mods), tracking={"n_steps": int(e.n_steps)},
                  native={FORMAT: native},
                  provenance=Provenance(format=FORMAT, original_name=str(e.name),
                                        original_type="FIELD_MAP"))
    if e.field_file is None:
        rep.lossy("FM_FILES_UNKNOWN", "field map has no source file prefix (built in memory)",
                  element=el.name, kind="FieldMap")
    elif fe:
        rep.lossy("FIELD_ERROR_NOT_CONVERTED", "HELIX per-seed cavity error dropped",
                  element=el.name, kind="FieldMap", **fe)
    else:
        rep.exact(el.name, "FieldMap")
    return el


def _convert_element(ctx: _FromCtx, e: Any, dE_eV: float) -> Element:
    mods, rep = ctx.mods, ctx.rep
    name = str(e.name)
    shift = _shift_p(e, mods)
    fe = _field_errors(e)

    if isinstance(e, mods.Drift):
        native = {"class": "Drift", "length_mm": float(e.length), "aperture": float(e.aperture),
                  "aperture_y": e.aperture_y, "x_shift": float(e.x_shift), "y_shift": float(e.y_shift)}
        rep.exact(name, "Drift")
        return Drift(name=name, length=float(e.length) * MM,
                     aperture=_aperture_p(e.aperture, e.aperture_y), shift=shift,
                     tracking={"n_steps": int(e.n_steps)}, native={FORMAT: native},
                     provenance=Provenance(format=FORMAT, original_name=name, original_type="DRIFT"))

    if isinstance(e, mods.Quadrupole):
        mp = MagneticMultipoleP(Bn={1: float(e.gradient)})
        if e.skew_angle:
            mp.tilt[1] = math.radians(float(e.skew_angle))
        for order, g in ((2, e.g3), (3, e.g4), (4, e.g5), (5, e.g6)):
            if g:
                mp.Bn[order] = float(g)
        native = {"class": "Quadrupole", "length_mm": float(e.length), "gfr": float(e.gfr),
                  "skew_angle": float(e.skew_angle), "aperture": float(e.aperture)}
        if fe:
            native["field_errors"] = fe
            rep.lossy("FIELD_ERROR_NOT_CONVERTED", "HELIX per-seed gradient error dropped",
                      element=name, kind="Quadrupole", **fe)
        else:
            rep.exact(name, "Quadrupole")
        return Quadrupole(name=name, length=float(e.length) * MM, multipole=mp,
                          aperture=_aperture_p(e.aperture), shift=shift,
                          tracking={"n_steps": int(e.n_steps)}, native={FORMAT: native},
                          provenance=Provenance(format=FORMAT, original_name=name,
                                                original_type="QUAD"))

    if isinstance(e, mods.Solenoid):
        native = {"class": "Solenoid", "length_mm": float(e.length), "aperture": float(e.aperture)}
        if fe:
            native["field_errors"] = fe
            rep.lossy("FIELD_ERROR_NOT_CONVERTED", "HELIX per-seed field error dropped",
                      element=name, kind="Solenoid", **fe)
        else:
            rep.exact(name, "Solenoid")
        return Solenoid(name=name, length=float(e.length) * MM,
                        solenoid=SolenoidP(Bsol_T=float(e.field)),
                        aperture=_aperture_p(e.aperture), shift=shift,
                        tracking={"n_steps": int(e.n_steps)}, native={FORMAT: native},
                        provenance=Provenance(format=FORMAT, original_name=name,
                                              original_type="SOLENOID"))

    if isinstance(e, mods.RFGap):
        native = {"class": "RFGap", "voltage_MV": float(e.voltage), "phase_deg": float(e.phase),
                  "frequency_mhz": float(e.frequency), "ttf": float(e.ttf),
                  "p_flag": int(e.p_flag), "aperture": float(e.aperture)}
        if fe:
            native["field_errors"] = fe
        rf = RFP(frequency_Hz=float(e.frequency) * MHZ,
                 voltage_V=float(e.voltage) * float(e.ttf) * MEV,
                 phase_rad=phase_from_tracewin_deg(float(e.phase), ctx.charge, False),
                 ttf=float(e.ttf), phase_is_sync=False)
        el = RFCavity(name=name, length=0.0, rf=rf, aperture=_aperture_p(e.aperture), shift=shift,
                      native={FORMAT: native},
                      provenance=Provenance(format=FORMAT, original_name=name, original_type="GAP"))
        if fe:
            rep.lossy("FIELD_ERROR_NOT_CONVERTED", "HELIX per-seed cavity error dropped",
                      element=name, kind="RFCavity", **fe)
        else:
            rep.equivalent("SYNC_MODE_ASSUMED",
                           "HELIX thin GAPs have no SET_SYNC_PHASE mode "
                           "(tracewin_parser.py:701); the deck phase is read as a RAW RF phase",
                           element=name, kind="RFCavity", pending_sync=ctx.pending_sync,
                           p_flag=int(e.p_flag))
        return el

    if isinstance(e, (mods.FieldMap, mods.FieldMap3D)):
        return _convert_fieldmap(ctx, e, dE_eV)

    if isinstance(e, mods.NCells):
        sync = bool(e.sync_phase)
        if sync:
            ctx.pending_sync = False
        params = {k: _jsonable(getattr(e, k)) for k in
                  ("mode", "n_cells", "beta_g", "eot_v_per_m", "theta_s_deg", "p_flag",
                   "k_eot_i", "k_eot_o", "dz_i_mm", "dz_o_mm", "frequency_mhz", "sync_phase",
                   "beta_s")}
        params["aperture_mm"] = float(e.aperture)
        ttf = getattr(e, "_ttf", None)
        if ttf is not None:
            params["ttf"] = {"beta_s": ttf.beta_s,
                             **{k: [getattr(ttf, k).Ts, getattr(ttf, k).kTp, getattr(ttf, k).k2Tpp]
                                for k in ("middle", "input", "output")}}
        rf = RFP(frequency_Hz=float(e.frequency_mhz) * MHZ,
                 phase_rad=phase_from_tracewin_deg(float(e.theta_s_deg), ctx.charge, sync),
                 dE_ref_eV=dE_eV, n_cell=int(e.n_cells), phase_is_sync=sync,
                 L_active_m=float(e.length) * MM)
        rep.exact(name, "NCells")
        return NCells(name=name, length=float(e.length) * MM, rf=rf, params=params,
                      aperture=_aperture_p(e.aperture), tracking={"n_steps": int(e.n_steps)},
                      native={FORMAT: {"class": "NCells", "length_mm": float(e.length)}},
                      provenance=Provenance(format=FORMAT, original_name=name,
                                            original_type="NCELLS"))

    if mods.RfqCell is not None and isinstance(e, mods.RfqCell):
        params = {k: _jsonable(getattr(e, k)) for k in
                  ("voltage_V", "r0_mm", "A10", "modulation", "phi_s_deg", "cell_type",
                   "Tc_mm", "dP_deg", "type_prev", "type_next", "field_model")}
        params["length_mm"] = float(e.length)
        params["aperture"] = float(e.aperture)
        params["A_quad"] = float(getattr(e, "_A_quad", 0.0))
        rf = RFP(voltage_V=float(e.voltage_V),
                 phase_rad=phase_from_tracewin_deg(float(e.phi_s_deg), ctx.charge, True),
                 dE_ref_eV=dE_eV, phase_is_sync=True, L_active_m=float(e.length) * MM)
        rep.exact(name, "RFQCell")
        return RFQCell(name=name, length=float(e.length) * MM, rf=rf, params=params,
                       aperture=_aperture_p(e.aperture), tracking={"n_steps": int(e.n_steps)},
                       native={FORMAT: {"class": "RfqCell", "length_mm": float(e.length)}},
                       provenance=Provenance(format=FORMAT, original_name=name,
                                             original_type="RFQ_CELL"))

    if mods.VaneRFQ is not None and isinstance(e, mods.VaneRFQ):
        rep.lossy("VANE_RFQ_GEOMETRY_NOT_CONVERTED",
                  "VaneRFQ vane geometry / cell spans have no IR representation",
                  element=name, kind="RFQCell", n_cells=len(getattr(e, "cells", []) or []))
        rf = RFP(dE_ref_eV=dE_eV, L_active_m=float(e.length) * MM)
        return RFQCell(name=name, length=float(e.length) * MM, rf=rf,
                       params={"helix_class": "VaneRFQ",
                               "field_model": str(getattr(e, "field_model", "")),
                               "n_cells": len(getattr(e, "cells", []) or [])},
                       aperture=_aperture_p(e.aperture), tracking={"n_steps": int(e.n_steps)},
                       native={FORMAT: {"class": "VaneRFQ", "length_mm": float(e.length)}},
                       provenance=Provenance(format=FORMAT, original_name=name,
                                             original_type="RFQ"))

    if isinstance(e, mods.SuperposedFieldMap):
        children = []
        kids = []
        for z0, child in e.children:
            child_ir = _convert_fieldmap(ctx, child, 0.0)
            kids.append((float(z0) * MM, child_ir))
        el = Superposition(name=name, length=float(e.length) * MM, children=children,
                           aperture=_aperture_p(e.aperture), shift=shift,
                           tracking={"n_steps": int(e.n_steps)},
                           native={FORMAT: {"class": "SuperposedFieldMap",
                                            "length_mm": float(e.length),
                                            "dE_ref_eV": dE_eV,
                                            "aperture": float(e.aperture)}},
                           provenance=Provenance(format=FORMAT, original_name=name,
                                                 original_type="SUPERPOSE_MAP"))
        ctx.superpositions.append((el, kids))
        rep.lossy("SUPERPOSE_DE_NOT_IN_WALK",
                  "IR Superposition has no RFP group, so walk() cannot apply its reference "
                  f"energy gain ({dE_eV:.6g} eV); kept in native['helix']['dE_ref_eV']",
                  element=name, kind="Superposition", dE_ref_eV=dE_eV)
        return el

    if isinstance(e, mods.Steerer):
        brho = ctx.ir_ref.brho_signed
        rigidity = brho * (ctx.ir_ref.beta * 299_792_458.0) if e.elec else brho
        native = {"class": "Steerer", "bx_l": float(e.bx_l), "by_l": float(e.by_l),
                  "elec": bool(e.elec)}
        if e.elec:
            hkick, vkick = float(e.bx_l) / rigidity, float(e.by_l) / rigidity
        else:
            hkick, vkick = float(e.by_l) / rigidity, float(e.bx_l) / rigidity
        rep.exact(name, "Kicker")
        return Kicker(name=name, hkick=hkick, vkick=vkick, electric=bool(e.elec),
                      native={FORMAT: native},
                      provenance=Provenance(format=FORMAT, original_name=name,
                                            original_type="THIN_STEERING"))

    if isinstance(e, mods.Aperture):
        t = int(e.aperture_type)
        if t == 1:
            ap = ApertureP.circle(float(e.dx) * MM)
        else:
            ap = ApertureP.rect(float(e.dx) * MM, float(e.dy) * MM)
        native = {"class": "Aperture", "dx": float(e.dx), "dy": float(e.dy), "aperture_type": t}
        if t in (2, 6):
            rep.lossy("APERTURE_TYPE_UNMODELLED",
                      f"TraceWin aperture type {t} (pepperpot / ring) has no IR shape",
                      element=name, kind="Collimator", aperture_type=t)
        else:
            rep.exact(name, "Collimator")
        return Collimator(name=name, aperture=ap, native={FORMAT: native},
                          provenance=Provenance(format=FORMAT, original_name=name,
                                                original_type="APERTURE"))

    if isinstance(e, mods.Multipole):
        mp = MagneticMultipoleP()
        brho = ctx.ir_ref.brho_signed
        for idx, k in enumerate(e.knl):
            if k:
                mp.BnL[idx + 1] = float(k) * brho
        for idx, k in enumerate(e.ksl):
            if k:
                mp.BsL[idx + 1] = float(k) * brho
        if e.tilt_deg:
            mp.tilt[1] = -math.radians(float(e.tilt_deg))   # HELIX tilt is MAD-X-reversed
        native = {"class": "Multipole", "knl": [float(k) for k in e.knl],
                  "ksl": [float(k) for k in e.ksl], "dx": float(e.dx), "dy": float(e.dy),
                  "tilt_deg": float(e.tilt_deg), "aperture": float(e.aperture),
                  "brho_signed": brho}
        rep.exact(name, "Multipole")
        return Multipole(name=name, multipole=mp, aperture=_aperture_p(e.aperture),
                         native={FORMAT: native},
                         provenance=Provenance(format=FORMAT, original_name=name,
                                               original_type="MULTIPOLE"))

    if isinstance(e, mods.ThinLens):
        M = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        if e.fx != float("inf"):
            M[1][0] = -1.0 / float(e.fx)
        if e.fy != float("inf"):
            M[3][2] = -1.0 / float(e.fy)
        rep.equivalent("THINLENS_AS_TAYLOR", "HELIX ThinLens stored as its 6x6 kick matrix",
                       element=name, kind="Taylor", fx=float(e.fx), fy=float(e.fy))
        return Taylor(name=name, matrix=M, aperture=_aperture_p(e.aperture),
                      native={FORMAT: {"class": "ThinLens", "fx": float(e.fx), "fy": float(e.fy),
                                       "aperture": float(e.aperture)}},
                      provenance=Provenance(format=FORMAT, original_name=name,
                                            original_type="THIN_LENS"))

    if isinstance(e, mods.MatrixElement):
        rep.equivalent("MATRIX_BASIS_HELIX",
                       "matrix kept in HELIX's (mm, mrad, mm, mrad, deg, MeV) basis",
                       element=name, kind="Taylor")
        return Taylor(name=name, length=float(e.length) * MM,
                      matrix=[[float(v) for v in row] for row in e.matrix],
                      offset=[0.0] * 6 if e.offset is None else [float(v) for v in e.offset],
                      aperture=_aperture_p(e.aperture),
                      native={FORMAT: {"class": "MatrixElement", "length_mm": float(e.length),
                                       "aperture": float(e.aperture)}},
                      provenance=Provenance(format=FORMAT, original_name=name,
                                            original_type="MATRIX"))

    if isinstance(e, mods.Foil):
        rep.exact(name, "Foil")
        return Foil(name=name, material=str(e.material),
                    thickness_kg_per_m2=float(e.thickness_ug_cm2) * 1e-5,
                    aperture=_aperture_p(e.aperture),
                    native={FORMAT: {"class": "Foil", "thickness_ug_cm2": float(e.thickness_ug_cm2),
                                     "seed": getattr(e, "seed", None),
                                     "straggling": str(getattr(e, "straggling", "auto")),
                                     "aperture": float(e.aperture)}},
                    provenance=Provenance(format=FORMAT, original_name=name, original_type="FOIL"))

    if isinstance(e, mods.SpaceChargeComp):
        rep.exact(name, "Directive")
        return Directive(name=name, format="tracewin", card="SPACE_CHARGE_COMP",
                         args=[repr(float(e.factor))], role="tracking",
                         native={FORMAT: {"class": "SpaceChargeComp", "factor": float(e.factor)}},
                         provenance=Provenance(format=FORMAT, original_name=name,
                                               original_type="SPACE_CHARGE_COMP"))

    if isinstance(e, mods.ScGridDirective):
        rep.exact(name, "Directive")
        return Directive(name=name, format="tracewin", card="HELIX_SC_GRID",
                         args=[repr(float(e.extent_sigma))], role="tracking",
                         native={FORMAT: {"class": "ScGridDirective",
                                          "extent_sigma": float(e.extent_sigma)}},
                         provenance=Provenance(format=FORMAT, original_name=name,
                                               original_type="HELIX_SC_GRID"))

    if isinstance(e, mods.Edge):
        native = {"class": "Edge", "pole_rotation": float(e.pole_rotation), "rho": float(e.rho),
                  "gap": float(e.gap), "k1": float(e.k1), "k2": float(e.k2),
                  "aperture": float(e.aperture_radius), "hv": int(e.hv)}
        rep.lossy("EDGE_WITHOUT_BEND",
                  "HELIX EDGE card with no adjacent BEND kept as a passthrough directive",
                  element=name, kind="Directive")
        return Directive(name=name, format="tracewin", card="EDGE",
                         args=[repr(native[k]) for k in ("pole_rotation", "rho", "gap", "k1", "k2")],
                         role="other", native={FORMAT: native},
                         provenance=Provenance(format=FORMAT, original_name=name,
                                               original_type="EDGE"))

    if isinstance(e, mods.Marker):
        return _convert_marker(ctx, e)

    if isinstance(e, mods.LatticeCommand):
        return _convert_command(ctx, e)

    rep.dropped("HELIX_ELEMENT_UNKNOWN",
                f"no IR mapping for HELIX element class {type(e).__name__}",
                element=name, kind="Marker", helix_class=type(e).__name__)
    return Marker(name=name, length=float(getattr(e, "length", 0.0)) * MM,
                  native={FORMAT: {"class": type(e).__name__}},
                  provenance=Provenance(format=FORMAT, original_name=name,
                                        original_type=type(e).__name__))


def from_helix(helix_lattice, *, species: str = "proton", kinetic_energy_eV: float = 2.1e6,
               frequency_Hz: float | None = None,
               name: str = "helix") -> tuple[Lattice, FidelityReport]:
    """Convert a ``linac_gen.core.lattice.Lattice`` into the lattix IR.

    The HELIX reference particle is advanced element by element exactly as
    ``linac_gen.tracking.matrix_tracking.compute_transfer_matrix`` does (see
    ``lattix/oracles/helix.py``), so field maps / NCELLS report the reference
    energy gain they really produce and every rigidity-dependent conversion
    uses the local Bρ.
    """
    mods = _helix()
    rep = FidelityReport(source_format=FORMAT, target_format="lattix")
    hspecies, sp_key = _helix_species(species, mods)
    freq_hz = frequency_Hz or _first_frequency_hz(helix_lattice, mods) or DEFAULT_FREQUENCY_HZ
    href = mods.Ref(hspecies, w_kin=kinetic_energy_eV / MEV, frequency=freq_hz / MHZ)
    ir_ref = ReferenceParticle(species=ir_species(sp_key), kinetic_energy_eV=float(kinetic_energy_eV),
                               rf_frequency_Hz=float(freq_hz))
    ctx = _FromCtx(mods=mods, rep=rep, charge=int(ir_ref.species.charge), ir_ref=ir_ref)

    src = list(helix_lattice.elements)
    out: list[Element] = []
    i = 0
    while i < len(src):
        e = src[i]
        if isinstance(e, mods.Dipole) or (
                isinstance(e, mods.Edge) and i + 1 < len(src) and isinstance(src[i + 1], mods.Dipole)):
            e_in, dip, e_out, n = _bend_group(src, i, mods)
            for member in (e_in, dip, e_out):
                if member is not None:
                    _advance_helix_ref(member, href, mods)
            ir_el = _convert_bend(ctx, e_in, dip, e_out)
            i += n
        else:
            w0 = href.w_kin
            _advance_helix_ref(e, href, mods)
            ir_el = _convert_element(ctx, e, (href.w_kin - w0) * MEV)
            i += 1
        out.append(ir_el)
        if isinstance(ir_el, Freq):
            ctx.ir_ref = ctx.ir_ref.advanced(rf_frequency_Hz=ir_el.frequency_Hz)
        else:
            ctx.ir_ref = ctx.ir_ref.advanced(dE_eV=_ir_energy_gain(ir_el, ctx.ir_ref),
                                             ds_m=ir_el.length)

    lat = Lattice.from_sequence(name, out, ir_ref)
    for sup, kids in ctx.superpositions:
        for z0, child in kids:
            sup.children.append((z0, lat.add_element(child)))
    lat.meta[FORMAT] = {
        "step_config": {"integration_steps_per_metre":
                        float(helix_lattice.step_config.integration_steps_per_metre),
                        "sc_steps_per_metre": float(helix_lattice.step_config.sc_steps_per_metre)},
        "rfq_geom_file": getattr(helix_lattice, "rfq_geom_file", None),
        "species": sp_key,
    }
    errs = {k: _jsonable(getattr(helix_lattice, k, None))
            for k in ("errors", "beam_errors", "error_ratios", "error_cutoff")}
    if errs["errors"] or errs["beam_errors"] or errs["error_ratios"] or errs["error_cutoff"]:
        lat.meta["helix_errors"] = errs
        rep.lossy("ERROR_STUDY_NOT_CONVERTED",
                  "HELIX ERROR_* study definitions kept in Lattice.meta['helix_errors'] only",
                  element=None, kind=None, n_errors=len(errs["errors"] or []),
                  n_beam_errors=len(errs["beam_errors"] or []),
                  error_ratios=errs["error_ratios"], error_cutoff=errs["error_cutoff"])
    return lat, rep


# ---------------------------------------------------------------------------
# to_helix
# ---------------------------------------------------------------------------
@dataclass
class _ToCtx:
    mods: Any
    rep: FidelityReport
    charge: int
    lattice: Lattice
    seen: set = field(default_factory=set)


def _nat(el: Element) -> dict:
    return el.native.get(FORMAT, {}) or {}


def _common_kwargs(el: Element, nat: dict) -> dict:
    """aperture + misalignment + n_steps kwargs shared by most HELIX constructors."""
    ap, _ = _aperture_mm(el.aperture, nat)
    kw: dict = {"aperture": ap}
    kw.update(_shift_kwargs(el.shift))
    n = el.tracking.get("n_steps")
    if n is not None:
        kw["n_steps"] = int(n)
    return kw


def _r_drift(ctx: _ToCtx, el: Element, ref) -> list:
    nat = _nat(el)
    ap, ap_y = _aperture_mm(el.aperture, nat)
    kw = _shift_kwargs(el.shift)
    xs, ys = float(nat.get("x_shift", 0.0)), float(nat.get("y_shift", 0.0))
    kw["dx"] = kw.get("dx", 0.0) - xs
    kw["dy"] = kw.get("dy", 0.0) - ys
    n = el.tracking.get("n_steps")
    ctx.rep.exact(el.name, "Drift")
    return [ctx.mods.Drift(el.name, length=_to_mm(el.length, nat.get("length_mm")), aperture=ap,
                           aperture_y=ap_y, x_shift=xs, y_shift=ys,
                           n_steps=int(n) if n is not None else 1, **kw)]


def _r_quad(ctx: _ToCtx, el: Element, ref) -> list:
    nat = _nat(el)
    mp = el.multipole
    kw = _common_kwargs(el, nat)
    kw.setdefault("n_steps", 5)
    ctx.rep.exact(el.name, "Quadrupole")
    return [ctx.mods.Quadrupole(
        el.name, length=_to_mm(el.length, nat.get("length_mm")), gradient=mp.Bn.get(1, 0.0),
        skew_angle=_to_deg(mp.tilt.get(1, 0.0), nat.get("skew_angle")),
        g3=mp.Bn.get(2, 0.0), g4=mp.Bn.get(3, 0.0), g5=mp.Bn.get(4, 0.0), g6=mp.Bn.get(5, 0.0),
        gfr=float(nat.get("gfr", 0.0)), **kw)]


def _r_pole(ctx: _ToCtx, el: Element, ref) -> list:
    """Sextupole / Octupole → HELIX Quadrupole with g3 / g4 (EQUIVALENT)."""
    nat = _nat(el)
    mp = el.multipole
    kw = _common_kwargs(el, nat)
    kw.setdefault("n_steps", 5)
    ctx.rep.equivalent("THICK_POLE_AS_QUAD_HIGHER_ORDER",
                       f"{el.kind} emitted as a HELIX QUAD with g3/g4 (thin split-operator kick)",
                       element=el.name, kind=el.kind)
    return [ctx.mods.Quadrupole(el.name, length=_to_mm(el.length, nat.get("length_mm")),
                                gradient=mp.Bn.get(1, 0.0), g3=mp.Bn.get(2, 0.0),
                                g4=mp.Bn.get(3, 0.0), g5=mp.Bn.get(4, 0.0),
                                g6=mp.Bn.get(5, 0.0), **kw)]


def _r_solenoid(ctx: _ToCtx, el: Element, ref) -> list:
    nat = _nat(el)
    kw = _common_kwargs(el, nat)
    kw.setdefault("n_steps", 5)
    ctx.rep.exact(el.name, "Solenoid")
    return [ctx.mods.Solenoid(el.name, length=_to_mm(el.length, nat.get("length_mm")),
                              field=el.solenoid.Bsol_T, **kw)]


def _r_bend(ctx: _ToCtx, el: Element, ref) -> list:
    mods, nat, b = ctx.mods, _nat(el), el.bend
    if b.angle == 0.0:
        ctx.rep.lossy("BEND_ZERO_ANGLE",
                      "HELIX Dipole length is |rho|·|angle|, so a zero-angle bend cannot keep "
                      "its length; emitted as a DRIFT", element=el.name, kind="Bend")
        return [mods.Drift(el.name, length=_to_mm(el.length, nat.get("length_mm")),
                           aperture=_aperture_mm(el.aperture, nat)[0])]
    angle_deg = _to_deg(b.angle, nat.get("angle_deg"))
    rho_mm = float(nat["rho_mm"]) if "rho_mm" in nat else abs(el.length / b.angle) / MM
    hv = 1 if abs(abs(b.tilt_ref) - math.pi / 2) < 1e-9 else 0
    if b.tilt_ref and not hv:
        ctx.rep.lossy("BEND_TILT_UNSUPPORTED",
                      "HELIX bends are horizontal (hv=0) or vertical (hv=1) only",
                      element=el.name, kind="Bend", tilt_ref=b.tilt_ref)
    ap, _ = _aperture_mm(el.aperture, nat)
    N = 0.0
    if el.multipole.Bn.get(1):
        rho_m = rho_mm * MM
        N = -el.multipole.Bn[1] * rho_m * rho_m / ref.brho_signed
    if "field_index" in nat and abs(N - float(nat["field_index"])) <= 1e-9 * max(
            1.0, abs(float(nat["field_index"]))):
        N = float(nat["field_index"])
    kw = _shift_kwargs(el.shift)
    n = el.tracking.get("n_steps")
    if n is not None:
        kw["n_steps"] = int(n)
    body = mods.Dipole(el.name, angle=angle_deg, rho=rho_mm, field_index=N,
                       aperture=ap, hv=hv, **kw)
    sgn = 1.0 if b.angle >= 0 else -1.0
    saved = nat.get("edges")
    out = []
    if saved is not None:
        for spec in saved[:1]:
            if spec is not None:
                out.append(mods.Edge(spec["name"], pole_rotation=spec["pole_rotation"],
                                     rho=spec["rho"], gap=spec["gap"], k1=spec["k1"],
                                     k2=spec["k2"], aperture=spec["aperture"], hv=spec["hv"]))
        out.append(body)
        for spec in saved[1:]:
            if spec is not None:
                out.append(mods.Edge(spec["name"], pole_rotation=spec["pole_rotation"],
                                     rho=spec["rho"], gap=spec["gap"], k1=spec["k1"],
                                     k2=spec["k2"], aperture=spec["aperture"], hv=spec["hv"]))
    elif b.e1 or b.e2 or b.hgap or b.edge_int1 or b.edge_int2:
        k2 = DEFAULT_FRINGE_K2 if b.fringe_k2 is None else b.fringe_k2
        gap_mm = 2.0 * b.hgap / MM
        out = [mods.Edge(f"{el.name}_e1", pole_rotation=sgn * math.degrees(b.e1), rho=rho_mm,
                         gap=gap_mm, k1=b.edge_int1, k2=k2, aperture=ap, hv=hv),
               body,
               mods.Edge(f"{el.name}_e2", pole_rotation=sgn * math.degrees(b.e2), rho=rho_mm,
                         gap=gap_mm, k1=b.edge_int1 if b.edge_int2 is None else b.edge_int2,
                         k2=k2, aperture=ap, hv=hv)]
    else:
        out = [body]
    ctx.rep.exact(el.name, "Bend")
    return out


def _r_rfcavity(ctx: _ToCtx, el: Element, ref) -> list:
    mods, nat, rf = ctx.mods, _nat(el), el.rf
    ttf = float(nat.get("ttf", rf.ttf or 1.0)) or 1.0
    volt_mv = float(nat["voltage_MV"]) if "voltage_MV" in nat and abs(
        rf.voltage_V - float(nat["voltage_MV"]) * ttf * MEV) <= 1e-9 * max(1.0, abs(rf.voltage_V)) \
        else rf.voltage_V / (ttf * MEV)
    freq_mhz = (rf.frequency_Hz or (ref.rf_frequency_Hz or DEFAULT_FREQUENCY_HZ)) / MHZ
    # HELIX thin GAPs have no SET_SYNC_PHASE mode, so the phase must be emitted RAW:
    # advance_ref uses dW = q·V·T·cos φ, which then reproduces the IR gain V_eff·cos φ_ir.
    phase_deg = float(nat["phase_deg"]) if "phase_deg" in nat and abs(
        phase_from_tracewin_deg(float(nat["phase_deg"]), ctx.charge, False) - rf.phase_rad
    ) <= 1e-12 else tracewin_phase_deg(rf.phase_rad, ctx.charge, False)
    kw = _shift_kwargs(el.shift)
    ap, _ = _aperture_mm(el.aperture, nat)
    gap = mods.RFGap(el.name, voltage=volt_mv, phase=phase_deg, frequency=freq_mhz, ttf=ttf,
                     aperture=ap, p_flag=int(nat.get("p_flag", 0)), **kw)
    if el.length > 0:
        half = _to_mm(el.length, nat.get("length_mm")) / 2.0
        ctx.rep.equivalent("THICK_CAVITY_SPLIT",
                           "thick RF cavity emitted as DRIFT L/2 + thin GAP + DRIFT L/2",
                           element=el.name, kind="RFCavity", length_m=el.length)
        return [mods.Drift(f"{el.name}_in", length=half, aperture=ap), gap,
                mods.Drift(f"{el.name}_out", length=half, aperture=ap)]
    if rf.phase_is_sync:
        ctx.rep.equivalent("SYNC_PHASE_AS_RAW",
                           "HELIX GAPs ignore SET_SYNC_PHASE (tracewin_parser.py:701); the IR "
                           "synchronous phase is emitted as the equivalent raw RF phase",
                           element=el.name, kind="RFCavity", phase_deg=phase_deg)
    else:
        ctx.rep.exact(el.name, "RFCavity")
    return [gap]


def _r_fieldmap(ctx: _ToCtx, el: Element, ref) -> list:
    mods, nat, rf = ctx.mods, _nat(el), el.rf
    prefix = (el.files or [None])[0] or nat.get("field_file")
    geom = el.geom if el.geom is not None else nat.get("geom")
    freq_mhz = (rf.frequency_Hz or DEFAULT_FREQUENCY_HZ) / MHZ
    n_steps = int(el.tracking.get("n_steps", nat.get("n_steps", 100)))
    length_mm = _to_mm(el.length, nat.get("length_mm"))
    ap, _ = _aperture_mm(el.aperture, nat)
    sync = int(el.p_flag) == 1 or rf.phase_is_sync
    phase_deg = float(nat["phase_deg"]) if "phase_deg" in nat and abs(
        phase_from_tracewin_deg(float(nat["phase_deg"]), ctx.charge, sync) - rf.phase_rad
    ) <= 1e-12 else tracewin_phase_deg(rf.phase_rad, ctx.charge, sync)
    if prefix is None or geom is None:
        ctx.rep.lossy("FM_FILES_MISSING", "field map has no file prefix / geom; emitted as a DRIFT",
                      element=el.name, kind="FieldMap", files=el.files, geom=geom)
        return [mods.Drift(el.name, length=length_mm, aperture=ap)]
    try:
        from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
        from linac_gen.io.tracewin_geom import decode_geom

        fdata = read_tracewin_fieldmap(geom=geom, prefix=prefix, base_dir=None,
                                       frequency=freq_mhz, Ki=el.ki, Ka=el.ka)
        code = decode_geom(geom)
    except Exception as exc:                                  # noqa: BLE001
        ctx.rep.lossy("FM_FILES_MISSING",
                      f"field map files for {prefix!r} unreadable ({exc}); emitted as a DRIFT",
                      element=el.name, kind="FieldMap", files=el.files, geom=geom)
        return [mods.Drift(el.name, length=length_mm, aperture=ap)]
    kw = _shift_kwargs(el.shift)
    fm = mods.make_field_map_element(
        name=el.name, code=code, length_mm=length_mm, field_data=fdata,
        kb=el.kb, ke=el.ke, ki=el.ki, ka=el.ka, phase=phase_deg, frequency=freq_mhz,
        aperture=ap, n_steps=n_steps, p_flag=int(el.p_flag), geom=geom, field_file=prefix)
    for k, v in kw.items():
        setattr(fm, k, v)
    ctx.rep.exact(el.name, "FieldMap")
    return [fm]


def _r_ncells(ctx: _ToCtx, el: Element, ref) -> list:
    mods, p, rf = ctx.mods, dict(el.params), el.rf
    ttf = None
    tt = p.pop("ttf", None)
    if isinstance(tt, dict):
        ttf = mods.TTFTable(beta_s=float(tt["beta_s"]),
                            **{k: mods.TTFSet(*[float(x) for x in tt[k]])
                               for k in ("middle", "input", "output")})
    sync = bool(p.get("sync_phase", rf.phase_is_sync))
    p["theta_s_deg"] = tracewin_phase_deg(rf.phase_rad, ctx.charge, sync)
    p.pop("beta_s", None)
    n = el.tracking.get("n_steps")
    try:
        cav = mods.NCells(el.name, ttf=ttf, n_steps=int(n) if n is not None else None,
                          **{k: p[k] for k in
                             ("mode", "n_cells", "beta_g", "eot_v_per_m", "theta_s_deg",
                              "aperture_mm", "p_flag", "k_eot_i", "k_eot_o", "dz_i_mm",
                              "dz_o_mm", "frequency_mhz", "sync_phase") if k in p})
    except Exception as exc:                                   # noqa: BLE001
        ctx.rep.lossy("NCELLS_PARAMS_MISSING",
                      f"cannot rebuild a HELIX NCELLS from params ({exc}); emitted as a DRIFT",
                      element=el.name, kind="NCells")
        return [mods.Drift(el.name, length=el.length / MM)]
    ctx.rep.exact(el.name, "NCells")
    return [cav]


def _r_rfqcell(ctx: _ToCtx, el: Element, ref) -> list:
    mods, p = ctx.mods, dict(el.params)
    if mods.RfqCell is None or "voltage_V" not in p:
        ctx.rep.lossy("RFQ_CELL_NOT_REBUILT",
                      "RFQCell parameters do not describe a HELIX RFQ_CELL; emitted as a DRIFT",
                      element=el.name, kind="RFQCell", params=sorted(p))
        return [mods.Drift(el.name, length=el.length / MM)]
    n = el.tracking.get("n_steps")
    ctx.rep.exact(el.name, "RFQCell")
    return [mods.RfqCell(el.name, voltage_V=p["voltage_V"], r0_mm=p["r0_mm"], A10=p["A10"],
                         modulation=p["modulation"],
                         length_mm=p.get("length_mm", el.length / MM),
                         phi_s_deg=p["phi_s_deg"], cell_type=p["cell_type"],
                         Tc_mm=p.get("Tc_mm", 0.0), dP_deg=p.get("dP_deg", 0.0),
                         n_steps=int(n) if n is not None else None,
                         type_prev=p.get("type_prev"), type_next=p.get("type_next"),
                         A_quad=p.get("A_quad"), aperture=p.get("aperture", 0.0),
                         field_model=p.get("field_model", "tw2term"))]


def _r_superposition(ctx: _ToCtx, el: Element, ref) -> list:
    mods, nat = ctx.mods, _nat(el)
    kids = []
    for z0, ref_name in el.children:
        child = ctx.lattice.elements.get(ref_name)
        if child is None:
            kids = []
            break
        built = _r_fieldmap(ctx, child, ref)
        if not built or not isinstance(built[0], (mods.FieldMap, mods.FieldMap3D)):
            kids = []
            break
        kids.append((z0 / MM, built[0]))
    if not kids:
        ctx.rep.lossy("SUPERPOSE_NOT_REBUILT",
                      "SUPERPOSE children unavailable; emitted as a DRIFT",
                      element=el.name, kind="Superposition")
        return [mods.Drift(el.name, length=_to_mm(el.length, nat.get("length_mm")))]
    n = el.tracking.get("n_steps")
    ctx.rep.exact(el.name, "Superposition")
    return [mods.SuperposedFieldMap(el.name, kids, aperture=nat.get("aperture"),
                                    n_steps=int(n) if n is not None else None,
                                    **_shift_kwargs(el.shift))]


def _r_kicker(ctx: _ToCtx, el: Element, ref) -> list:
    nat = _nat(el)
    rigidity = ref.brho_signed
    if el.electric:
        rigidity *= ref.beta * 299_792_458.0
        bx, by = el.hkick * rigidity, el.vkick * rigidity
    else:
        by, bx = el.hkick * rigidity, el.vkick * rigidity
    if "bx_l" in nat:
        bx = float(nat["bx_l"]) if abs(bx - float(nat["bx_l"])) <= 1e-9 * max(1.0, abs(bx)) else bx
        by = float(nat["by_l"]) if abs(by - float(nat["by_l"])) <= 1e-9 * max(1.0, abs(by)) else by
    ctx.rep.exact(el.name, "Kicker")
    return [ctx.mods.Steerer(el.name, bx_l=bx, by_l=by, elec=bool(el.electric))]


def _r_collimator(ctx: _ToCtx, el: Element, ref) -> list:
    nat = _nat(el)
    if "dx" in nat:
        dx, dy, t = float(nat["dx"]), float(nat["dy"]), int(nat["aperture_type"])
    else:
        ap = el.aperture
        t = 1 if (ap is None or ap.shape == "ELLIPTICAL") else 0
        dx = 0.0 if ap is None or ap.half_x is None else ap.half_x / MM
        dy = dx if ap is None or ap.half_y is None else ap.half_y / MM
    ctx.rep.exact(el.name, "Collimator")
    return [ctx.mods.Aperture(el.name, dx=dx, dy=dy, aperture_type=t)]


def _r_marker(ctx: _ToCtx, el: Element, ref) -> list:
    nat = _nat(el)
    params = getattr(el, "params", {}) or {}
    mk = ctx.mods.Marker(
        el.name, snapshot=bool(nat.get("snapshot", getattr(el, "family", "") == "DIAG_PHASE")),
        is_bpm=bool(nat.get("is_bpm", getattr(el, "family", "") == "BPM")),
        diag_family=nat.get("diag_family", params.get("diag_family")),
        x_target_mm=nat.get("x_target_mm", params.get("x_target_mm")),
        y_target_mm=nat.get("y_target_mm", params.get("y_target_mm")),
        accuracy_mm=nat.get("accuracy_mm", params.get("accuracy_mm", 1.0)),
        origin_keyword=nat.get("origin_keyword"))
    for extra in ("lattice_card_args", "origin_params"):
        if extra in nat:
            setattr(mk, extra, nat[extra])
    ctx.rep.exact(el.name, el.kind)
    return [mk]


def _r_foil(ctx: _ToCtx, el: Element, ref) -> list:
    nat = _nat(el)
    ug = float(nat["thickness_ug_cm2"]) if "thickness_ug_cm2" in nat \
        else el.thickness_kg_per_m2 / 1e-5
    ctx.rep.exact(el.name, "Foil")
    return [ctx.mods.Foil(el.name, material=el.material, thickness_ug_cm2=ug,
                          aperture=_aperture_mm(el.aperture, nat)[0],
                          seed=nat.get("seed"), straggling=nat.get("straggling", "auto"))]


def _r_taylor(ctx: _ToCtx, el: Element, ref) -> list:
    mods, nat = ctx.mods, _nat(el)
    if nat.get("class") == "ThinLens":
        ctx.rep.equivalent("THINLENS_AS_TAYLOR", "Taylor restored to a HELIX ThinLens",
                           element=el.name, kind="Taylor")
        return [mods.ThinLens(el.name, fx=float(nat["fx"]), fy=float(nat["fy"]),
                              aperture=float(nat.get("aperture", 0.0)))]
    offset = None if not any(el.offset) else list(el.offset)
    ctx.rep.equivalent("MATRIX_BASIS_HELIX",
                       "matrix written verbatim into HELIX's (mm, mrad, deg, MeV) basis",
                       element=el.name, kind="Taylor")
    return [mods.MatrixElement(el.name, matrix=el.matrix,
                               length=_to_mm(el.length, nat.get("length_mm")), offset=offset,
                               aperture=_aperture_mm(el.aperture, nat)[0])]


def _r_multipole(ctx: _ToCtx, el: Element, ref) -> list:
    nat, mp = _nat(el), el.multipole
    brho = ref.brho_signed
    order = max([0, *mp.BnL, *mp.BsL])
    knl = [mp.BnL.get(n, 0.0) / brho for n in range(1, order + 1)]
    ksl = [mp.BsL.get(n, 0.0) / brho for n in range(1, order + 1)]
    if "knl" in nat and len(nat["knl"]) >= len(knl) and all(
            abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(knl, nat["knl"], strict=False)):
        knl, ksl = list(nat["knl"]), list(nat["ksl"])
    ctx.rep.exact(el.name, "Multipole")
    return [ctx.mods.Multipole(el.name, knl=knl, ksl=ksl,
                               aperture=_aperture_mm(el.aperture, nat)[0],
                               dx=float(nat.get("dx", 0.0)), dy=float(nat.get("dy", 0.0)),
                               tilt_deg=_to_deg(-mp.tilt.get(1, 0.0), nat.get("tilt_deg")))]


def _r_reference_change(ctx: _ToCtx, el: Element, ref) -> list:
    mods, nat = ctx.mods, _nat(el)
    kw = nat.get("kwargs", {})
    if el.energy_eV is not None:
        ctx.rep.exact(el.name, "ReferenceChange")
        return [mods.SetBeamEnergy(el.name, k=int(kw.get("k", 0)), energy_MeV=el.energy_eV / MEV)]
    ctx.rep.exact(el.name, "ReferenceChange")
    return [mods.SetBeamE0P0(el.name, k=int(kw.get("k", 0)),
                             dE_MeV=(el.dE_ref_eV or 0.0) / MEV,
                             dphi_deg=float(kw.get("dphi_deg", 0.0)),
                             ke=int(kw.get("ke", 1 if el.dE_ref_eV else 0)),
                             kp=int(kw.get("kp", 0)))]


def _r_freq(ctx: _ToCtx, el: Element, ref) -> list:
    ctx.rep.exact(el.name, "Freq")
    return [ctx.mods.Freq(el.name, frequency_mhz=el.frequency_Hz / MHZ)]


def _r_directive(ctx: _ToCtx, el: Element, ref) -> list:
    mods, nat = ctx.mods, _nat(el)
    card = el.card or nat.get("card", "")
    cls_name = nat.get("class")
    if cls_name == "SpaceChargeComp" or card == "SPACE_CHARGE_COMP":
        ctx.rep.exact(el.name, "Directive")
        return [mods.SpaceChargeComp(el.name, factor=float(nat.get("factor", el.args[0])))]
    if cls_name == "ScGridDirective" or card == "HELIX_SC_GRID":
        ctx.rep.exact(el.name, "Directive")
        return [mods.ScGridDirective(el.name, extent_sigma=float(nat.get("extent_sigma",
                                                                        el.args[0])))]
    if cls_name == "Edge" or card == "EDGE":
        ctx.rep.lossy("EDGE_WITHOUT_BEND", "stand-alone EDGE restored from the passthrough",
                      element=el.name, kind="Directive")
        return [mods.Edge(el.name, pole_rotation=float(nat["pole_rotation"]),
                          rho=float(nat["rho"]), gap=float(nat["gap"]), k1=float(nat["k1"]),
                          k2=float(nat["k2"]), aperture=float(nat.get("aperture", 0.0)),
                          hv=int(nat.get("hv", 0)))]
    cmd = _rebuild_command(mods, card, el, nat)
    if cmd is not None:
        ctx.rep.exact(el.name, "Directive")
        return [cmd]
    ctx.rep.lossy("DIRECTIVE_DROPPED",
                  f"no HELIX LatticeCommand for {el.format}:{card!r}; emitted as a MARKER",
                  element=el.name, kind="Directive", card=card, args=el.args)
    return [mods.Marker(el.name, snapshot=False)]


def _rebuild_command(mods, card: str, el: Element, nat: dict):
    cls = mods.COMMAND_CLASSES.get(card)
    if cls is None:
        return mods.Freq(el.name, frequency_mhz=float(el.args[0])) if card == "FREQ" else None
    if card.startswith("ADJUST_BEAM_"):
        args = list(el.args) or [nat.get("kwargs", {}).get("diag_n", 0)]
        flags = nat.get("kwargs", {}).get("flags") or [int(float(a)) for a in args[1:]]
        return cls(el.name, int(float(args[0])), *flags)
    kwargs = nat.get("kwargs")
    if kwargs:
        try:
            return cls(el.name, **kwargs)
        except TypeError:
            pass
    schema = mods.SCHEMA.get(card)
    if schema is not None:
        try:
            return cls(el.name, **mods.parse_positionals(schema, list(el.args)))
        except Exception:                                      # noqa: BLE001
            pass
    try:
        return cls(el.name)
    except Exception:                                          # noqa: BLE001
        return None


def _r_patch(ctx: _ToCtx, el: Element, ref) -> list:
    ctx.rep.lossy("PATCH_NOT_SUPPORTED",
                  "HELIX has no reference-frame patch; emitted as a MARKER",
                  element=el.name, kind="Patch")
    return [ctx.mods.Marker(el.name, snapshot=False)]


RULES: dict[str, Any] = {
    "Drift": _r_drift,
    "Quadrupole": _r_quad,
    "Sextupole": _r_pole,
    "Octupole": _r_pole,
    "Multipole": _r_multipole,
    "Bend": _r_bend,
    "Solenoid": _r_solenoid,
    "RFCavity": _r_rfcavity,
    "FieldMap": _r_fieldmap,
    "NCells": _r_ncells,
    "RFQCell": _r_rfqcell,
    "Kicker": _r_kicker,
    "Collimator": _r_collimator,
    "Marker": _r_marker,
    "Instrument": _r_marker,
    "Foil": _r_foil,
    "Taylor": _r_taylor,
    "Patch": _r_patch,
    "ReferenceChange": _r_reference_change,
    "Freq": _r_freq,
    "Directive": _r_directive,
    "Superposition": _r_superposition,
}


def to_helix(lattice: Lattice, *, species: str | None = None) -> tuple[Any, FidelityReport]:
    """Convert a lattix IR lattice into a ``linac_gen.core.lattice.Lattice``."""
    mods = _helix()
    rep = FidelityReport(source_format="lattix", target_format=FORMAT)
    sp = species or lattice.meta.get(FORMAT, {}).get("species") or lattice.reference.species.name
    _helix_species(sp, mods)          # validate early
    charge = int(ir_species(sp).charge)
    hlat = mods.Lattice()
    ctx = _ToCtx(mods=mods, rep=rep, charge=charge, lattice=lattice)

    warnings: list[str] = []
    child_names = {n for e in lattice.elements.values() if isinstance(e, Superposition)
                   for _, n in e.children}
    for p in propagate(lattice, warnings=warnings):
        el = p.element
        if el.name in child_names:
            continue
        rule = RULES.get(el.kind)
        if rule is None:                                       # pragma: no cover - RULES is total
            rep.dropped("KIND_NOT_MAPPED", f"no HELIX rule for IR kind {el.kind}",
                        element=el.name, kind=el.kind)
            continue
        for built in rule(ctx, el, p.ref_in or lattice.reference):
            hlat.add(built)

    meta = lattice.meta.get(FORMAT, {})
    sc = meta.get("step_config")
    if sc:
        from linac_gen.core.step_config import StepConfig

        hlat.step_config = StepConfig(**sc)
    if meta.get("rfq_geom_file"):
        hlat.rfq_geom_file = meta["rfq_geom_file"]
    if lattice.meta.get("helix_errors"):
        rep.lossy("ERROR_STUDY_NOT_CONVERTED",
                  "Lattice.meta['helix_errors'] is a record only; no ErrorDef objects rebuilt",
                  element=None, kind=None)
    for w in warnings:
        rep.lossy("WALK_WARNING", w, element=None, kind=None)
    return hlat, rep


# ---------------------------------------------------------------------------
# convenience: parse a deck with HELIX's own readers, then convert
# ---------------------------------------------------------------------------
_PARSERS = {".dat": "tracewin", ".madx": "madx", ".seq": "madx", ".mad": "madx",
            ".lat": "mad8", ".flat": "mad8", ".lte": "elegant"}


def read_helix_deck(path: str | Path, **kw) -> tuple[Lattice, FidelityReport]:
    """Parse *path* with HELIX's own reader and convert the result to the IR.

    ``kw`` is forwarded to :func:`from_helix`; ``species`` /
    ``kinetic_energy_eV`` / ``frequency_Hz`` default to whatever the HELIX
    parser reports in its metadata.
    """
    _helix()
    p = Path(path)
    fmt = _PARSERS.get(p.suffix.lower(), "tracewin")
    if fmt == "madx":
        from linac_gen.io.madx_parser import parse_madx as _parse
    elif fmt == "mad8":
        from linac_gen.io.mad8_parser import parse_mad8 as _parse
    elif fmt == "elegant":
        from linac_gen.io.elegant_parser import parse_elegant as _parse
    else:
        from linac_gen.io.tracewin_parser import parse_tracewin as _parse
    hlat, meta = _parse(str(p))[:2]
    ref = meta.get("reference") if isinstance(meta, dict) else None
    if ref is not None:
        kw.setdefault("kinetic_energy_eV", float(ref.w_kin) * MEV)
        kw.setdefault("frequency_Hz", float(ref.frequency) * MHZ)
        kw.setdefault("species", "h-" if int(ref.species.charge) < 0 else
                      ("deuteron" if float(ref.species.mass) > 1500 else "proton"))
    kw.setdefault("name", p.stem)
    lat, rep = from_helix(hlat, **kw)
    rep.source_file = str(p)
    lat.warnings.extend(str(w) for w in (meta.get("warnings", []) if isinstance(meta, dict) else []))
    return lat, rep
