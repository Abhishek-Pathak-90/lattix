"""Element kinds and parameter groups (names mirror the PALS standard where it has them).

Conventions (PLAN §4.1):
* SI + eV; lengths m, angles rad, fields T, gradients T/m, voltages V, frequencies Hz;
* canonical strengths are LAB fields (``MagneticMultipoleP.Bn[1]`` = G [T/m]); normalized
  views (K1 = G/Bρ_signed) live in :mod:`lattix.ir.normalize`;
* ``RFP.phase_rad`` is the SYNCHRONOUS phase for the reference particle, cos convention,
  0 = crest, so the reference energy gain is ``voltage_V · cos(phase_rad)`` regardless of
  species (MAD-X, Bmad and PALS semantics; TraceWin raw RF phases and Elegant phases are
  charge-signed and converted by their readers);
* every parameter is a float; the source expression (``k1 := kqf``) is kept in
  ``Element.expressions[param]`` for writers that support variables;
* ``native[format]`` is an opaque passthrough re-emitted only to the same format.
"""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from lattix.ir.expr import Expression


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Provenance(_Model):
    format: str
    file: str | None = None
    line: int | None = None
    original_name: str | None = None
    original_type: str | None = None


class ApertureP(_Model):
    shape: Literal["ELLIPTICAL", "RECTANGULAR"] = "ELLIPTICAL"
    x_limits: tuple[float, float] | None = None     # m, (min, max)
    y_limits: tuple[float, float] | None = None
    aperture_at: Literal["ENTRANCE", "EXIT", "BOTH_ENDS", "CONTINUOUS"] = "BOTH_ENDS"

    @classmethod
    def circle(cls, radius_m: float) -> ApertureP:
        return cls(shape="ELLIPTICAL", x_limits=(-radius_m, radius_m), y_limits=(-radius_m, radius_m))

    @classmethod
    def rect(cls, half_x_m: float, half_y_m: float) -> ApertureP:
        return cls(shape="RECTANGULAR", x_limits=(-half_x_m, half_x_m), y_limits=(-half_y_m, half_y_m))

    @property
    def half_x(self) -> float | None:
        return None if self.x_limits is None else (self.x_limits[1] - self.x_limits[0]) / 2

    @property
    def half_y(self) -> float | None:
        return None if self.y_limits is None else (self.y_limits[1] - self.y_limits[0]) / 2


class BodyShiftP(_Model):
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.0
    x_rot: float = 0.0     # rad (pitch)
    y_rot: float = 0.0     # rad (yaw)
    tilt: float = 0.0      # rad (roll about s)

    def is_zero(self) -> bool:
        return not any((self.x_offset, self.y_offset, self.z_offset, self.x_rot, self.y_rot, self.tilt))


class MagneticMultipoleP(_Model):
    """Lab-frame multipole content.  ``Bn[n]``/``Bs[n]`` are per-length field
    derivatives of order n (n = 0 dipole [T], 1 gradient [T/m], 2 [T/m²] …);
    ``BnL``/``BsL`` are integrated (thin) strengths [T·m^(1−n)].  ``tilt[n]`` in rad."""

    Bn: dict[int, float] = Field(default_factory=dict)
    Bs: dict[int, float] = Field(default_factory=dict)
    BnL: dict[int, float] = Field(default_factory=dict)
    BsL: dict[int, float] = Field(default_factory=dict)
    tilt: dict[int, float] = Field(default_factory=dict)

    def is_zero(self) -> bool:
        return not any(v for d in (self.Bn, self.Bs, self.BnL, self.BsL) for v in d.values())


class BendP(_Model):
    # PALS spells `angle` as `angle_ref` and stores edge integrals as the product fint·hgap
    angle: float = 0.0            # rad, signed; horizontal plane unless tilt_ref
    e1: float = 0.0               # rad, sector-referenced entrance pole-face angle
    e2: float = 0.0
    edge_int1: float = 0.0        # fint
    edge_int2: float | None = None   # fintx (None = same as edge_int1)
    hgap: float = 0.0             # m, half gap
    tilt_ref: float = 0.0         # rad; ±π/2 = vertical bend
    rect: bool = False            # source was an RBEND (e1/e2 already include angle/2)
    fringe_k2: float | None = None   # TraceWin EDGE K2 (2.80 default there)

    def g_ref(self, length: float) -> float:
        return 0.0 if length == 0 else self.angle / length


class RFP(_Model):
    frequency_Hz: float | None = None
    harmon: float | None = None       # PALS/MAD harmonic number when the deck gives it instead of a frequency
    voltage_V: float = 0.0            # effective accelerating voltage for the reference particle (≥ 0 normally)
    gradient_V_per_m: float | None = None
    phase_rad: float = 0.0            # synchronous phase, 0 = crest, cos convention
    L_active_m: float | None = None
    n_cell: int | None = None
    ttf: float = 1.0                  # informational: TraceWin GAP folds T into E0TL
    dE_ref_eV: float | None = None    # explicit reference energy change (field maps, PALS)
    cavity_type: Literal["STANDING_WAVE", "TRAVELING_WAVE"] = "STANDING_WAVE"
    phase_is_sync: bool = True        # False: phase_rad is the deck's entrance RF phase for the reference
                                      # particle (species-normalised), not a synchronous phase; the walk
                                      # still applies V·cos(phase) as its best estimate of the gain


class SolenoidP(_Model):
    Bsol_T: float = 0.0


# ---------------------------------------------------------------------------
class Element(_Model):
    kind: ClassVar[str] = "Element"
    name: str
    length: float = 0.0
    aperture: ApertureP | None = None
    shift: BodyShiftP | None = None
    tracking: dict = Field(default_factory=dict)          # n_steps, integrator hints (PALS TrackingP)
    native: dict[str, dict] = Field(default_factory=dict)  # {format: {...}} passthrough
    expressions: dict[str, Expression] = Field(default_factory=dict)
    provenance: Provenance | None = None
    meta: dict = Field(default_factory=dict)

    @property
    def is_thin(self) -> bool:
        return self.length == 0.0


class Drift(Element):
    kind = "Drift"


class Quadrupole(Element):
    kind = "Quadrupole"
    multipole: MagneticMultipoleP = Field(default_factory=MagneticMultipoleP)

    @property
    def gradient(self) -> float:          # T/m, positive focuses x for q > 0
        return self.multipole.Bn.get(1, 0.0)

    @gradient.setter
    def gradient(self, g: float) -> None:
        self.multipole.Bn[1] = g

    @property
    def skew_rad(self) -> float:
        return self.multipole.tilt.get(1, 0.0)


class Sextupole(Element):
    kind = "Sextupole"
    multipole: MagneticMultipoleP = Field(default_factory=MagneticMultipoleP)   # Bn[2] [T/m²]


class Octupole(Element):
    kind = "Octupole"
    multipole: MagneticMultipoleP = Field(default_factory=MagneticMultipoleP)   # Bn[3]


class Multipole(Element):
    """Thin multipole: integrated strengths ``BnL``/``BsL``."""
    kind = "Multipole"
    multipole: MagneticMultipoleP = Field(default_factory=MagneticMultipoleP)


class Bend(Element):
    kind = "Bend"
    bend: BendP = Field(default_factory=BendP)
    multipole: MagneticMultipoleP = Field(default_factory=MagneticMultipoleP)   # Bn[1]: gradient (field index)

    @property
    def rho(self) -> float | None:
        return None if self.bend.angle == 0 else self.length / self.bend.angle


class Solenoid(Element):
    kind = "Solenoid"
    solenoid: SolenoidP = Field(default_factory=SolenoidP)


class RFCavity(Element):
    """Thin gap (length 0, TraceWin GAP) or thick cavity (Bmad lcavity, Elegant RFCA)."""
    kind = "RFCavity"
    rf: RFP = Field(default_factory=RFP)


class FieldMap(Element):
    """TraceWin geom-coded field map (channels × dimension, several files)."""
    kind = "FieldMap"
    rf: RFP = Field(default_factory=RFP)
    geom: int | None = None
    files: list[str] = Field(default_factory=list)     # base name + resolved paths in meta
    ke: float = 1.0                                     # electric amplitude scale
    kb: float = 1.0                                     # magnetic amplitude scale
    ki: float = 0.0
    ka: float = 1.0
    p_flag: int = 0


class NCells(Element):
    """TraceWin NCELLS (DTL/CCL cell train); parameters kept verbatim in ``params``."""
    kind = "NCells"
    rf: RFP = Field(default_factory=RFP)
    params: dict = Field(default_factory=dict)


class RFQCell(Element):
    kind = "RFQCell"
    rf: RFP = Field(default_factory=RFP)
    params: dict = Field(default_factory=dict)


class Kicker(Element):
    """Orbit corrector: canonical deflections of the reference particle [rad]."""
    kind = "Kicker"
    hkick: float = 0.0
    vkick: float = 0.0
    electric: bool = False


class Collimator(Element):
    """Thin aperture restriction (TraceWin APERTURE card, MAD-X *COLLIMATOR)."""
    kind = "Collimator"


class Marker(Element):
    kind = "Marker"


class Instrument(Element):
    """Diagnostic hardware / monitor; ``family`` e.g. BPM, PROFILE, CURRENT."""
    kind = "Instrument"
    family: str = "MONITOR"
    params: dict = Field(default_factory=dict)


class Foil(Element):
    kind = "Foil"
    material: str = "C"
    thickness_kg_per_m2: float = 0.0
    dE_ref_eV: float | None = None      # PALS FoilP.dE_ref (reference energy loss)


class Taylor(Element):
    """Explicit first-order map (MAD-X MATRIX, Elegant EMATRIX, Bmad taylor order 1)."""
    kind = "Taylor"
    matrix: list[list[float]] = Field(default_factory=lambda: [[1.0 if i == j else 0.0 for j in range(6)]
                                                                for i in range(6)])
    offset: list[float] = Field(default_factory=lambda: [0.0] * 6)
    basis: Literal["madx", "elegant", "bmad", "xtrack", "tracewin", "flame", "impactx", "cheetah", "common"] | None = None  # noqa: E501  # source coordinate basis


class Patch(Element):
    kind = "Patch"
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.0
    x_rot: float = 0.0
    y_rot: float = 0.0
    tilt: float = 0.0
    t_offset_s: float | None = None        # Bmad patch t_offset
    e_tot_offset_eV: float | None = None   # Bmad patch e_tot_offset


class ReferenceChange(Element):
    """Adjust the reference energy / phase (TraceWin SET_BEAM_ENERGY / SET_BEAM_E0_P0)."""
    kind = "ReferenceChange"
    dE_ref_eV: float | None = None      # relative change
    energy_eV: float | None = None      # or an absolute kinetic energy
    dphase_rad: float | None = None     # reference phase shift w.r.t. the RF clock (SET_BEAM_E0_P0 kp)
    dtime_s: float | None = None        # equivalent time shift


class Freq(Element):
    """TraceWin FREQ card: the machine RF clock switches here."""
    kind = "Freq"
    frequency_Hz: float = 0.0


class Directive(Element):
    """Format-specific zero-length command kept inline (TraceWin SET_*, ADJUST*, LATTICE,
    ERROR_*, DIAG targets …).  ``role`` classifies it; writers of other formats drop it
    with a DROPPED entry unless they can map the role."""
    kind = "Directive"
    format: str = "tracewin"
    card: str = ""
    args: list[str] = Field(default_factory=list)
    role: str = "other"     # matching | period_start | period_end | error | sync_phase | diag | other


class Superposition(Element):
    """Overlapping field maps (TraceWin SUPERPOSE_MAP cluster); children are element
    names with their z offsets [m] from the cluster start.  ``rf.dE_ref_eV`` carries the
    cluster's reference energy gain so the walk can apply it."""
    kind = "Superposition"
    children: list[tuple[float, str]] = Field(default_factory=list)
    rf: RFP = Field(default_factory=RFP)
    ref: str | None = None            # Bmad superimpose ref element (when not resolved geometrically)
    offset_m: float | None = None     # Bmad superimpose offset


ELEMENT_KINDS: dict[str, type[Element]] = {
    c.kind: c for c in (Drift, Quadrupole, Sextupole, Octupole, Multipole, Bend, Solenoid, RFCavity,
                        FieldMap, NCells, RFQCell, Kicker, Collimator, Marker, Instrument, Foil,
                        Taylor, Patch, ReferenceChange, Freq, Directive, Superposition)
}
ALL_KINDS: frozenset[str] = frozenset(ELEMENT_KINDS)
RF_KINDS: frozenset[str] = frozenset({"RFCavity", "FieldMap", "NCells", "RFQCell"})


def element_to_dict(e: Element) -> dict:
    d = e.model_dump(mode="json")
    d["kind"] = e.kind
    return d


def element_from_dict(d: dict) -> Element:
    d = dict(d)
    cls = ELEMENT_KINDS[d.pop("kind")]
    return cls.model_validate(d)
