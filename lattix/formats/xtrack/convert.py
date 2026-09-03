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

``energy_mode="local"`` (default)
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
import re
import warnings
from dataclasses import dataclass
from typing import Any

from lattix._version import __version__
from lattix.fidelity import FidelityReport
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
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.units import C_LIGHT
from lattix.ir.walk import energy_gain_eV, propagate

#: an element counts as accelerating above 1 µeV of reference gain (same tolerance as
#: the MAD-X writer: far below any physical gain, far above ``cos(±π/2)`` noise).
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


def to_line(lattice: Lattice, *, energy_mode: str = "local", report: FidelityReport | None = None,
            name: str | None = None, strict: bool = False, install_apertures: bool = True):
    """Build an :class:`xtrack.Line` from an IR lattice, in flat order.

    Parameters
    ----------
    energy_mode:
        ``"local"`` (default) normalizes every strength with Bρ at that element's
        entrance; ``"constant"`` uses the lattice start.  Either way the choice is
        recorded in the fidelity ledger (xtrack keeps p0c fixed through RF).
    report:
        optional :class:`~lattix.fidelity.FidelityReport` to fill; a fresh one is used
        when omitted (retrieve it through :class:`lattix.formats.xtrack.Writer`).
    install_apertures:
        emit ``LimitRect``/``LimitEllipse`` elements for :class:`ApertureP` data
        attached to ordinary elements (a :class:`Collimator` always becomes one).
    """
    import xtrack as xt

    if energy_mode not in ("local", "constant"):
        raise ValueError(f"energy_mode must be 'local' or 'constant', got {energy_mode!r}")
    rep = report if report is not None else FidelityReport()
    rep.target_format = "xtrack"

    placed = propagate(lattice)
    start_brho = lattice.reference.brho_signed
    b = _Builder(rep)

    for group, p in enumerate(placed):
        el = p.element
        ref: ReferenceParticle = p.ref_in or lattice.reference
        brho = ref.brho_signed if energy_mode == "local" else start_brho
        _emit(b, el, brho, ref, lattice, group, rep, install_apertures)

    line = xt.Line(elements=b.elements, element_names=list(b.order))
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


def _record_energy_mode(placed: list[Placed], energy_mode: str, rep: FidelityReport) -> None:
    local = energy_mode == "local"
    code = "CONST_P0_LOCAL_RIGIDITY" if local else "CONST_P0_START_RIGIDITY"
    msg = ("normalized strengths use the local rigidity at each element's entrance"
           if local else "normalized strengths use the rigidity at the start of the lattice")
    for p in placed:
        ref = p.ref_in
        if ref is None:
            continue
        dE = energy_gain_eV(p.element, ref)
        if abs(dE) <= _ACCEL_TOL_eV:
            continue
        rep.equivalent(code, msg, element=p.element.name, kind=p.element.kind,
                       dE_eV=dE, brho=ref.brho_signed)


def _record(rep: FidelityReport, el: Element, rule: Rule, **details) -> None:
    if rule.cls == "EXACT":
        rep.exact(el.name, el.kind)
    else:
        rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind,
                target=rule.target, **details)


def _emit(b: _Builder, el: Element, brho: float, ref: ReferenceParticle, lattice: Lattice,
          group: int, rep: FidelityReport, install_apertures: bool) -> None:
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
            _emit(b, child, brho, ref, lattice, group, rep, install_apertures)
        return

    if isinstance(el, Patch):
        _emit_patch(b, el, group, rep)
        _record(rep, el, rule)
        return

    made = _build(el, brho, ref, rep)
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


def _cavity(el, brho, ref, rep, *, voltage: float, length: float):
    import xtrack as xt

    rf = el.rf
    kw: dict[str, Any] = {"voltage": voltage, "phase": xtrack_phase_rad(rf.phase_rad)}
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


def _build(el: Element, brho: float, ref: ReferenceParticle, rep: FidelityReport):
    """``(main xtrack element, [extra elements])`` for one IR element."""
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
        return _cavity(el, brho, ref, rep, voltage=volt, length=el.length), []

    if isinstance(el, FieldMap):
        rf = el.rf
        volt = rf.voltage_V
        if not volt and rf.gradient_V_per_m is not None:
            volt = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        if not volt and rf.dE_ref_eV:
            volt = rf.dE_ref_eV / max(math.cos(rf.phase_rad), 1e-12)
        if volt:
            return _cavity(el, brho, ref, rep, voltage=volt, length=el.length), []
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
        kw = {"knl": [-el.hkick], "ksl": [el.vkick], "length": el.length}
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

        return xt.FirstOrderTaylorMap(length=el.length,
                                      m0=np.asarray(el.offset, dtype=float),
                                      m1=np.asarray(el.matrix, dtype=float)), []

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
    import xtrack as xt

    cls = getattr(xt, d.pop("__class__"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# xt.Line -> IR
# ---------------------------------------------------------------------------
#: :func:`_convert_element` maps these xtrack classes onto IR kinds; anything else becomes
#: a Marker with a ``DROPPED:UNSUPPORTED_XTRACK_ELEMENT`` entry and a verbatim ``native``
#: passthrough that :func:`to_line` re-emits unchanged.
KNOWN_CLASSES = frozenset({
    "Drift", "DriftExact", "Quadrupole", "Sextupole", "Octupole", "Multipole", "Bend",
    "RBend", "UniformSolenoid", "Solenoid", "Cavity", "Marker", "LimitRect", "LimitEllipse",
    "XYShift", "SRotation", "XRotation", "YRotation", "ZetaShift", "FirstOrderTaylorMap",
    "ReferenceEnergyIncrease",
})


def _particle_ref_reference(line, warn: list[str]) -> ReferenceParticle | None:
    import numpy as np

    pref = getattr(line, "particle_ref", None)
    if pref is None:
        return None
    mass_eV = float(np.atleast_1d(pref.mass0)[0])
    charge = int(round(float(np.atleast_1d(pref.q0)[0])))
    ke = float(np.atleast_1d(pref.kinetic_energy0)[0])
    nm = _species_name(mass_eV, charge)
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


def from_line(line, reference: ReferenceParticle | None = None, *,
              report: FidelityReport | None = None, name: str | None = None,
              energy_mode: str | None = None) -> Lattice:
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
            continue

        el, norm = _convert_element(raw, xname, row, rep, warn)
        if el is None:
            continue
        # aperture data and a thick collimator's drift live on sibling elements
        _attach_group_extras(el, names, rows, elements, group)
        out.append(el)
        scaled.append((el, norm))

    lat_name = name or meta.get("lattice") or getattr(line, "name", None) or "xtrack_line"
    lat = Lattice.from_sequence(lat_name, out, ref)
    lat.meta["source_format"] = "xtrack"
    if meta:
        lat.meta["xtrack"] = {k: v for k, v in meta.items() if k != "elements"}
    lat.warnings.extend(warn)

    # -- second pass: local rigidity ---------------------------------------
    mode = energy_mode or meta.get("energy_mode") or "local"
    placed = propagate(lat)
    start_brho = ref.brho_signed
    by_id = {id(e): n for e, n in scaled}
    for p in placed:
        norm = by_id.get(id(p.element))
        if not norm:
            continue
        brho = (p.ref_in or ref).brho_signed if mode == "local" else start_brho
        _apply_rigidity(p.element, norm, brho)
    for w in warn:
        if w not in lat.warnings:                          # pragma: no cover - defensive
            lat.warnings.append(w)
    return lat


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
    prov = Provenance(format="xtrack", original_name=row.get("original_name") or xname,
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
        el = Bend(name=ir_name, length=length,
                  bend=BendP(angle=h * length,
                             e1=float(raw.edge_entry_angle), e2=float(raw.edge_exit_angle),
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
    sh = BodyShiftP(x_offset=float(raw.shift_x), y_offset=float(raw.shift_y),
                    z_offset=float(raw.shift_s), tilt=float(raw.rot_s_rad_no_frame),
                    x_rot=float(raw.rot_x_rad), y_rot=float(raw.rot_y_rad))
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
