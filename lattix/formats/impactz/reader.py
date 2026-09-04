"""IMPACT-Z ``ImpactZ.in`` reader.

Everything below was read out of the IMPACT-Z sources at
``particle_tracking_codes/tier1_linac/IMPACT-Z`` (version 2.7.1 banner, conda-forge
``impact-z`` 2.7.7 binary) and, where marked MEASURED, confirmed by running
``ImpactZexe`` on this machine 2026-09-03.

File shape (``src/Contrl/Input.f90:63-197``, ``in1_Input``/``in2_Input``)
------------------------------------------------------------------------
A line whose first list-directed item is ``!`` is a comment and is skipped, so the
eleven header records are the first eleven *non-comment* lines:

===  ====================================================================
 1   ``npcol nprow``                    processor grid (product = # MPI ranks)
 2   ``dim np flagmap flagerr flagdiag``  6 = 6-D; np particles per charge state;
     ``flagmap`` 1 = linear-map integrator, 2 = nonlinear Lorentz integrator;
     ``flagerr`` 1 = apply the per-element misalignment columns;
     ``flagdiag`` 1 = standard diagnostics (fort.18/24…32), 2 = + 99 % emittance
 3   ``nx ny nz flagbc xrad yrad perdlen``  space-charge mesh, boundary code,
     pipe half-widths [m] and the longitudinal period [m]
 4   ``flagdist rstartflg flagsbstp nchrg``  distribution type, restart flag,
     space-charge sub-cycling flag, number of charge states
 5   ``nptlist(1:nchrg)``               particles per charge state
 6   ``currlist(1:nchrg)``              current per charge state [A]
 7   ``qmcclist(1:nchrg)``              q_i/m_i [1/eV] (charge in units of e,
     mass in eV; ``-1/511005`` for an electron in ``Example2``)
 8   ``distparam(1:7)``                 x: alpha, beta [m], eps_n [m·rad],
     xscale, pxscale, xmu1, xmu2
 9   ``distparam(8:14)``                y: same
10   ``distparam(15:21)``               z: alpha, beta [deg/MeV], eps [deg·MeV],
     zscale, pzscale, xmu5, xmu6
11   ``current energy mass charge frequency phase``  bunched-beam current [A],
     reference KINETIC energy [eV], rest mass [eV], charge in units of the proton
     charge, RF reference frequency [Hz], initial reference phase [rad]
===  ====================================================================

``Input.f90:181`` counts elements by reading ``length nseg mapstp type`` off every
non-comment line until ``type == -99``; ``in2_Input`` (``Input.f90:300-312``) then
reads ``length nseg mapstp type value1 … value24``.  Trailing columns that the
line does not supply keep the value 0 (list-directed input stops at ``/``), so a
deck writes only what it needs and ends each card with ``/``.

Element parameters (``AccSimulator.f90:237-520`` copies ``value_k`` into
``Param(k+1)``; ``Param(1)`` is the runtime ``zedge``)
-------------------------------------------------------------------------------
======  ==========  ==================================================================
 code    class      value1, value2, …  (units)
======  ==========  ==================================================================
   0    DriftTube   radius [m]                                   (``DriftTube.f90:20``)
   1    Quadrupole  gradient [T/m], file ID, radius [m], dx, dy, rot_x, rot_y, rot_z
                    (``Quadrupole.f90:20-28``; file ID > 1e-5 loads ``rfdataN.in``
                    as G(z), ``AccSimulator.f90:847-858``)
   2    ConstFoc    kx0² , ky0², kz0² [1/m²], radius             (``ConstFoc.f90:20-24``)
   3    Solenoid    Bz0 [T], file ID, radius [m], dx, dy, rot…    (``Sol.f90:20-28``)
   4    Dipole      angle [rad], k1 [1/m²], file ID, hgap [m], e1 [rad], e2 [rad],
                    h1 [1/m], h2 [1/m], fint, dx, dy, rot…
                    (``Dipole.f90:20-30`` names 2,3,4,5 and 11-15; 6..10 read off the
                    Transport set-up in ``AccSimulator.f90:1046-1060``: ``ang0``,
                    ``hd1``, ``angF``, ``angB``, ``hF``, ``hB``, ``dstr1``, with
                    ``hgap = 2·Param(5)`` so ``Param(5)`` is the HALF gap and
                    ``dstr1`` is MAD's ``fint``.  Angles are radians — the
                    ``*pi/180`` conversions are commented out.)
   5    Multipole   id (2 sext, 3 oct, 4 dec), field strength [T/m^n], file ID,
                    radius, dx, dy, rot…                        (``Multipole.f90:21-29``)
                    ``getfld_Multipole`` uses the MAD/Wiedemann convention
                    ``By = B_n·(x²−y²)/2!`` … so the strength is the IR ``Bn[n]``.
   6    Wiggler     id (1 planar, 2 helical), max field [T], file ID, radius, kx,
                    period [m], dx, dy, rot…                    (``Wiggler.f90:20-31``)
 101    DTL         scale, frequency [Hz], theta0 [deg], file ID, radius,
                    quad1 length [m], quad1 gradient [T/m], quad2 length, quad2
                    gradient, then 3×5 misalignment blocks      (``DTL.f90:20-44``)
 102    CCDTL       scale, frequency [Hz], theta0 [deg], file ID, radius, dx, dy, rot…
 103    CCL         same as 102                                 (``CCL.f90:20-30``)
 104    SC cavity   same as 102                                 (``SC.f90:20-30``)
 105    SolRF       … + Bz0 [T], aawk, ggwk, lengwk             (``SolRF.f90:20-34``)
 106    TWS         … + theta1 [deg], aawk, ggwk, lengwk        (``TWS.f90:23-37``)
 110    EMfld       scale, frequency, theta0, file ID, x radius, y radius, dx, dy,
                    rot…, data flag, coordinate flag           (``EMfld.f90:22-36``)
======  ==========  ==================================================================

Negative codes are "BPM" pseudo-elements (``BPM.f90:22-38``, dispatched in
``AccSimulator.f90:875-1016``); ``Param(k+1) = value_k`` there too, so the ``drange(3)``
the dispatcher reads is ``value2``:

======  ================================================================================
 code   meaning
======  ================================================================================
   -1   shift the transverse centroid to 0
   -2   dump the 9-column phase space to ``fort.<mapstp>``; ``value1`` = sample period
   -3   accumulated 1-D density   -4  1-D density   -5  2-D density   -6  3-D density
   -7   dump particles + geometry to ``fort.<mapstp>+rank``
   -8   slice information; ``value1`` = # slices, then alpha_x, beta_x, alpha_y, beta_y
  -10   mismatch the beam by factors in ``value2 … value7``
  -13   collimator slit: ``value2 … value5`` = x_min, x_max, y_min, y_max [m]
  -21   shift the 6-D centroid: ``value2 … value7`` = dx [m], dpx [rad], dy [m],
        dpy [rad], dz [deg], dPz [MeV].  ``kick_BPM`` (``BPM.f90:238-246``) adds
        ``dpx·γβ_z`` to p_x, so dpx/dpy are deflection ANGLES, and it adds
        ``dPz·1e6/mass`` to the 6th coordinate, which is ``γ0 − γ``: a positive
        ``dPz`` therefore *lowers* the energy.
  -40   RF nonlinearity kick: ``value2`` V [V], ``value3`` phase [deg], ``value4`` harmonic
  -41   read a discrete wakefield        -52  laser heater
  -55   thin-lens multipole: ``value2 … value7`` = k0 … k5
  -99   end of the lattice
======  ================================================================================

Internal coordinates (MEASURED 2026-09-03 — see :mod:`lattix.oracles.impactz`)
------------------------------------------------------------------------------
``Pts1(1:6) = (x/λ̄, γβ_x, y/λ̄, γβ_y, ω(t − t_ref) [rad], γ0 − γ)`` with
``λ̄ = Scxl = c/(2π f_ref)`` (``PhysConst.f90:30``).  The 5th coordinate is
*late-positive* and the 6th is the *negative* energy deviation.

Fidelity
--------
The reader keeps every raw column in ``element.native["impactz"]`` so an
IMPACT-Z → IMPACT-Z round trip re-emits unsupported cards byte-for-byte, and
records a ledger entry for every card it downgrades.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

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
    FieldMap,
    Foil,
    Instrument,
    Kicker,
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
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, Species, species
from lattix.ir.units import C_LIGHT

#: type code -> the IMPACT-Z class that implements it (documentation / ``native``).
TYPE_NAMES: dict[int, str] = {
    0: "drift", 1: "quadrupole", 2: "constfoc", 3: "solenoid", 4: "dipole",
    5: "multipole", 6: "wiggler", 101: "dtl", 102: "ccdtl", 103: "ccl",
    104: "sc_cavity", 105: "solrf", 106: "tws", 110: "emfld",
    -1: "bpm_center", -2: "phase_dump", -3: "dens1d_acc", -4: "dens1d", -5: "dens2d",
    -6: "dens3d", -7: "restart_dump", -8: "slice_info", -10: "mismatch",
    -13: "collimator_slit", -21: "centroid_shift", -40: "rf_kick", -41: "wakefield",
    -52: "laser_heater", -55: "thin_multipole", -99: "end",
}

#: RF-like codes: ``Param(5)`` (= ``value4``) is the ``rfdataN.in`` file id
#: (``AccSimulator.f90:818-820``).
RF_TYPES: frozenset[int] = frozenset({101, 102, 103, 104, 105, 106, 110})

#: ``Multipole.f90:21`` id column -> multipole order n (``Bn[n]`` in T/m^n).
MULTIPOLE_ORDER: dict[int, int] = {2: 2, 3: 3, 4: 4}

_TAG = re.compile(r"lattix:\s*(.*)$")
_KV = re.compile(r"(\w+)=(\S+)")
_FORTRAN_EXP = re.compile(r"(?<=[\d.])[dD]([-+]?\d)")


def _to_float(tok: str) -> float:
    """Fortran real: ``1.3d9`` and ``1.0E+3`` both parse."""
    return float(_FORTRAN_EXP.sub(r"e\1", tok))


def _is_comment(line: str) -> bool:
    """IMPACT-Z skips a record whose first list-directed item is ``!``
    (``Input.f90:69-73``): in practice a line whose first non-blank character is ``!``."""
    s = line.strip()
    return not s or s.startswith("!")


@dataclass
class Card:
    """One beam-line element line: ``length nseg mapstp type value1 … /``."""

    length: float
    nseg: int
    mapstp: int
    itype: int
    values: list[float] = field(default_factory=list)
    line: int = 0
    comment: str = ""          # text after the terminating "/"
    tag: dict[str, str] = field(default_factory=dict)   # decoded "! lattix:" tag

    def v(self, i: int) -> float:
        """``value_i`` (1-based), 0.0 when the card does not supply it."""
        return self.values[i - 1] if 0 < i <= len(self.values) else 0.0

    @property
    def name(self) -> str:
        return TYPE_NAMES.get(self.itype, f"type{self.itype}")


@dataclass
class Header:
    """The eleven header records of an ``ImpactZ.in``."""

    npcol: int = 1
    nprow: int = 1
    dim: int = 6
    np: int = 1000
    flagmap: int = 1
    flagerr: int = 0
    flagdiag: int = 1
    nx: int = 64
    ny: int = 64
    nz: int = 64
    flagbc: int = 1
    xrad: float = 0.014
    yrad: float = 0.014
    perdlen: float = 0.1
    flagdist: int = 3
    rstartflg: int = 0
    flagsbstp: int = 0
    nchrg: int = 1
    nptlist: list[int] = field(default_factory=lambda: [1000])
    currlist: list[float] = field(default_factory=lambda: [0.0])
    qmcclist: list[float] = field(default_factory=lambda: [0.0])
    distparam: list[float] = field(default_factory=lambda: [0.0] * 21)
    current_A: float = 0.0
    kinetic_energy_eV: float = 0.0
    mass_eV: float = 0.0
    charge: float = 1.0
    frequency_Hz: float = 0.0
    phase_ini_rad: float = 0.0

    @property
    def scxl(self) -> float:
        """``Scxl = c/(2π f)`` [m] — the internal transverse length scale
        (``PhysConst.f90:30``)."""
        return C_LIGHT / (2.0 * math.pi * self.frequency_Hz) if self.frequency_Hz else 0.0


def split_data_line(line: str) -> tuple[list[str], str]:
    """Fortran list-directed input: ``/`` ends the record, the rest is free text."""
    body, sep, rest = line.partition("/")
    return body.replace(",", " ").split(), (rest.strip() if sep else "")


def parse_deck(text: str) -> tuple[Header, list[Card], list[str]]:
    """Split an ``ImpactZ.in`` into its header, its element cards and its comments.

    Raises ``ValueError`` when the eleven header records are incomplete.
    """
    raw = text.splitlines()
    pending_tag: dict[str, str] = {}
    records: list[tuple[int, str]] = []
    tags: dict[int, dict[str, str]] = {}
    comments: list[str] = []
    for i, line in enumerate(raw, start=1):
        if _is_comment(line):
            comments.append(line.rstrip())
            m = _TAG.search(line)
            if m:
                pending_tag = dict(_KV.findall(m.group(1)))
            continue
        records.append((i, line))
        if pending_tag:
            tags[len(records) - 1] = pending_tag
            pending_tag = {}

    if len(records) < 11:
        raise ValueError(f"ImpactZ.in needs 11 header records, found {len(records)}")
    h = Header()
    nums: list[list[float]] = []
    for _, line in records[:11]:
        toks, _ = split_data_line(line)
        nums.append([_to_float(t) for t in toks])

    def need(row: int, n: int) -> list[float]:
        if len(nums[row]) < n:
            raise ValueError(f"ImpactZ.in header line {row + 1} needs {n} numbers, "
                             f"got {len(nums[row])}")
        return nums[row]

    h.npcol, h.nprow = (int(v) for v in need(0, 2)[:2])
    r = need(1, 5)
    h.dim, h.np, h.flagmap, h.flagerr, h.flagdiag = (int(v) for v in r[:5])
    r = need(2, 7)
    h.nx, h.ny, h.nz, h.flagbc = (int(v) for v in r[:4])
    h.xrad, h.yrad, h.perdlen = r[4:7]
    r = need(3, 4)
    h.flagdist, h.rstartflg, h.flagsbstp, h.nchrg = (int(v) for v in r[:4])
    n = max(1, h.nchrg)
    h.nptlist = [int(v) for v in need(4, n)[:n]]
    h.currlist = need(5, n)[:n]
    h.qmcclist = need(6, n)[:n]
    dist: list[float] = []
    for row in (7, 8, 9):
        vals = need(row, 7)[:7]
        dist.extend(vals)
    h.distparam = dist
    r = need(10, 6)
    (h.current_A, h.kinetic_energy_eV, h.mass_eV, h.charge, h.frequency_Hz,
     h.phase_ini_rad) = r[:6]

    cards: list[Card] = []
    for idx, (lineno, line) in enumerate(records[11:], start=11):
        toks, rest = split_data_line(line)
        if len(toks) < 4:
            raise ValueError(f"line {lineno}: an IMPACT-Z element card needs at least "
                             f"'length nseg mapstp type', got {line.strip()!r}")
        vals = [_to_float(t) for t in toks]
        c = Card(length=vals[0], nseg=int(vals[1]), mapstp=int(vals[2]), itype=int(vals[3]),
                 values=vals[4:], line=lineno, comment=rest, tag=tags.get(idx, {}))
        cards.append(c)
        if c.itype == -99:
            break
    return h, cards, comments


def species_from_header(h: Header) -> Species:
    """Best-matching :class:`Species` for the header's ``mass``/``charge``.

    IMPACT-Z carries no species name, so the mass is matched against the IR table
    to 1e-6 relative and otherwise a custom species is built.
    """
    q = int(round(h.charge)) or 1
    for nm in ("proton", "h-", "electron", "positron", "deuteron", "antiproton"):
        sp = species(nm)
        if sp.charge == q and abs(sp.mass_eV - h.mass_eV) <= 1e-6 * sp.mass_eV:
            return sp
    return Species(name=f"impactz_m{h.mass_eV:.6g}_q{q}", mass_eV=h.mass_eV, charge=q)


def rfdata_name(file_id: int) -> str:
    """``rfdataN.in`` / ``rfdataNN.in`` / ``rfdataNNN.in`` (``Data.f90:149-176``)."""
    return f"rfdata{int(file_id)}.in"


def read_rfdata(path: Path) -> list[list[float]]:
    """Numbers of an ``rfdataN.in``, one list per line (1 column = Fourier
    coefficients for ``read3_Data``, 4 columns = ``z, Ez, Ez', Ez''`` for
    ``read1_Data``)."""
    out: list[list[float]] = []
    for line in path.read_text(errors="replace").splitlines():
        toks, _ = split_data_line(line)
        if not toks or line.lstrip().startswith("!"):
            continue
        try:
            out.append([_to_float(t) for t in toks])
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "impactz"

    def read(self, path: Path, *, encoding: str = "utf-8",
             load_rfdata: bool = True, **_options) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="impactz", source_file=str(path))
        header, cards, _comments = parse_deck(path.read_text(encoding=encoding,
                                                             errors="replace"))
        sp = species_from_header(header)
        ref = ReferenceParticle(species=sp, kinetic_energy_eV=header.kinetic_energy_eV,
                                rf_frequency_Hz=header.frequency_Hz or None)
        self.rep = rep
        self.header = header
        self.dir = path.parent
        self.load_rfdata = load_rfdata
        self._counts: dict[str, int] = {}

        elements: list[Element] = []
        for card in cards:
            if card.itype == -99:
                break
            el = self._element(card)
            if el is not None:
                elements.append(self._promote(el, card))

        lat = Lattice.from_sequence(path.stem or "impactz", elements, ref)
        lat.meta.update({
            "source_format": "impactz",
            "impactz_header": _header_meta(header),
        })
        if header.current_A:
            rep.lossy("IMPACTZ_BEAM_CURRENT",
                      f"deck current {header.current_A:g} A (space charge) is not part of the IR",
                      details={"current_A": header.current_A})
        return lat, rep

    # -- helpers ----------------------------------------------------------
    #: kinds the writer degrades to a plain drift; a ``! lattix: kind=`` tag restores
    #: them so that IMPACT-Z ``→`` IR ``→`` IMPACT-Z is a fixed point (PLAN I-13/I-15).
    PROMOTIONS: dict[str, type[Element]] = {
        "Marker": Marker, "Instrument": Instrument, "Foil": Foil, "Taylor": Taylor,
        "Patch": Patch, "RFQCell": RFQCell, "Superposition": Superposition,
        "ReferenceChange": ReferenceChange,
    }

    def _promote(self, el: Element, card: Card) -> Element:
        """Rebuild a lattix-written element that had to be degraded to a drift."""
        want = card.tag.get("kind")
        cls = self.PROMOTIONS.get(want or "")
        if cls is None or el.kind != "Drift" or want == el.kind:
            return el
        out = cls(name=el.name, length=el.length, aperture=el.aperture,
                  native=dict(el.native), tracking=dict(el.tracking),
                  provenance=el.provenance, meta=dict(el.meta))
        self.rep.exact(el.name, want, code="IMPACTZ_NAME_TAG",
                       message=f"restored from the '! lattix:' tag ({want})")
        return out

    def _name(self, card: Card, stem: str) -> str:
        if "name" in card.tag:
            return unquote(card.tag["name"])
        self._counts[stem] = self._counts.get(stem, 0) + 1
        return f"{stem}_{self._counts[stem]}"

    @staticmethod
    def _aperture(radius: float) -> ApertureP | None:
        return ApertureP.circle(radius) if radius > 0 else None

    @staticmethod
    def _shift(card: Card, first: int) -> BodyShiftP | None:
        """``dx, dy, rot_x, rot_y, rot_z`` starting at ``value<first>``.
        Applied by IMPACT-Z only when the header's ``flagerr`` is 1."""
        vals = [card.v(first + k) for k in range(5)]
        if not any(vals):
            return None
        return BodyShiftP(x_offset=vals[0], y_offset=vals[1], x_rot=vals[2],
                          y_rot=vals[3], tilt=vals[4])

    def _native(self, card: Card, *, passthrough: bool = False) -> dict:
        d = {"type": card.itype, "class": card.name, "nseg": card.nseg,
             "mapstp": card.mapstp, "values": list(card.values), "line": card.line}
        if card.comment:
            d["comment"] = card.comment
        if passthrough:
            d["passthrough"] = True
        return d

    def _common(self, el: Element, card: Card, *, passthrough: bool = False) -> Element:
        el.native["impactz"] = self._native(card, passthrough=passthrough)
        el.tracking.setdefault("n_steps", card.nseg)
        el.tracking.setdefault("map_steps", card.mapstp)
        el.provenance = Provenance(format="impactz", line=card.line,
                                   original_type=card.name)
        return el

    # -- dispatch ---------------------------------------------------------
    def _element(self, card: Card) -> Element | None:
        t = card.itype
        fn = getattr(self, f"_t{t}" if t >= 0 else f"_tm{-t}", None)
        if fn is None:
            return self._unsupported(card)
        return fn(card)

    def _unsupported(self, card: Card) -> Element:
        """Unknown or unmodelled type code: keep the geometry, drop the physics."""
        name = self._name(card, TYPE_NAMES.get(card.itype, "type"))
        el: Element = (Drift(name=name, length=card.length) if card.length > 0
                       else Marker(name=name))
        message = (f"IMPACT-Z type {card.itype} ({card.name}) has no IR kind; "
                   f"kept as a {el.kind} with the raw columns in native['impactz']")
        if card.itype in TYPE_NAMES:      # a type IMPACT-Z documents but the IR does not model
            self.rep.add("LOSSY", "UNMODELLED_IMPACTZ_TYPE", message,
                         element=name, kind=el.kind, line=card.line, type=card.itype)
        else:                             # a type code unknown to IMPACT-Z itself
            self.rep.add("DROPPED", "UNSUPPORTED_IMPACTZ_TYPE", message,
                         element=name, kind=el.kind, line=card.line, type=card.itype)
        return self._common(el, card, passthrough=True)

    # -- positive type codes ---------------------------------------------
    def _t0(self, card: Card) -> Element:
        el = Drift(name=self._name(card, "drift"), length=card.length,
                   aperture=self._aperture(card.v(1)))
        return self._common(el, card)

    def _t1(self, card: Card) -> Element:
        el = Quadrupole(name=self._name(card, "quad"), length=card.length,
                        aperture=self._aperture(card.v(3)), shift=self._shift(card, 4))
        el.multipole.Bn[1] = card.v(1)
        if card.v(2) > 1e-5:      # AccSimulator.f90:847 loads rfdataN.in as G(z)
            self.rep.lossy("IMPACTZ_QUAD_GRADIENT_PROFILE",
                           f"quadrupole reads its gradient profile from "
                           f"{rfdata_name(int(card.v(2)))}; the IR keeps the hard-edge "
                           "gradient only", element=el.name, kind="Quadrupole",
                           line=card.line, file_id=int(card.v(2)))
        return self._common(el, card)

    def _t3(self, card: Card) -> Element:
        el = Solenoid(name=self._name(card, "sol"), length=card.length,
                      aperture=self._aperture(card.v(3)), shift=self._shift(card, 4),
                      solenoid=SolenoidP(Bsol_T=card.v(1)))
        return self._common(el, card)

    def _t4(self, card: Card) -> Element:
        hgap = card.v(4)
        el = Bend(name=self._name(card, "bend"), length=card.length,
                  aperture=self._aperture(hgap), shift=self._shift(card, 10))
        el.bend = BendP(angle=card.v(1), e1=card.v(5), e2=card.v(6),
                        edge_int1=card.v(9), hgap=hgap)
        if card.v(2):
            # AccSimulator.f90:1051 hd1 = Param(3) is the normalized quadrupole
            # component k1 [1/m²]; the IR stores the lab gradient.
            el.multipole.Bn[1] = card.v(2) * _brho(self.header)
        if card.v(7) or card.v(8):
            self.rep.lossy("IMPACTZ_POLE_FACE_CURVATURE",
                           "pole-face curvatures h1/h2 have no IR field",
                           element=el.name, kind="Bend", line=card.line,
                           h1=card.v(7), h2=card.v(8))
        return self._common(el, card)

    def _t5(self, card: Card) -> Element:
        order = MULTIPOLE_ORDER.get(int(round(card.v(1))))
        name = self._name(card, "mult")
        if order is None:
            return self._unsupported(card)
        cls = {2: Sextupole, 3: Octupole}.get(order, Multipole)
        el = cls(name=name, length=card.length, aperture=self._aperture(card.v(4)),
                 shift=self._shift(card, 5))
        el.multipole.Bn[order] = card.v(2)
        if card.v(3) > 1e-5:
            self.rep.lossy("IMPACTZ_MULTIPOLE_PROFILE",
                           f"multipole reads its profile from {rfdata_name(int(card.v(3)))}",
                           element=name, kind=el.kind, line=card.line)
        return self._common(el, card)

    def _rf_element(self, card: Card, *, kind: str) -> Element:
        """101/102/103/104/105/106/110: ``scale, frequency, theta0, file ID, radius…``.

        A **negative file ID** selects IMPACT-Z's ideal-cavity model
        (``BeamBunch.f90:323-437``): ``scale`` is then the accelerating gradient [V/m],
        ``theta0`` a *synchronous* phase [deg] and the reference gain is
        ``scale·L·cos θ0`` independently of the charge — the IR's own RF rule.
        A non-negative file ID means a real on-axis profile in ``rfdataN.in``.
        """
        name = self._name(card, kind)
        file_id = int(round(card.v(4)))
        ideal = card.v(4) < 0.0
        freq = card.v(2) or None
        rf = RFP(frequency_Hz=freq, phase_rad=math.radians(card.v(3)),
                 phase_is_sync=ideal,     # theta0 is a driven RF phase unless ideal (SC.f90:145)
                 L_active_m=card.length)
        if ideal:
            rf.gradient_V_per_m = card.v(1)
            rf.voltage_V = card.v(1) * card.length
        el: Element
        if card.itype == 101 and not ideal:
            el = NCells(name=name, length=card.length, rf=rf,
                        aperture=self._aperture(card.v(5)), shift=self._shift(card, 20))
            el.params = {"quad1_length_m": card.v(6), "quad1_gradient_T_per_m": card.v(7),
                         "quad2_length_m": card.v(8), "quad2_gradient_T_per_m": card.v(9)}
        elif card.itype in (101, 102, 103):
            el = NCells(name=name, length=card.length, rf=rf,
                        aperture=self._aperture(card.v(5)),
                        shift=self._shift(card, 20 if card.itype == 101 else 6))
        elif ideal:
            el = RFCavity(name=name, length=card.length, rf=rf,
                          aperture=self._aperture(card.v(5)), shift=self._shift(card, 6))
        else:
            el = FieldMap(name=name, length=card.length, rf=rf, ke=card.v(1),
                          aperture=self._aperture(card.v(5)), shift=self._shift(card, 6))
            if card.itype == 105:
                el.meta["impactz_Bz0_T"] = card.v(11)
        el.meta["impactz_scale"] = card.v(1)
        el.meta["impactz_file_id"] = file_id
        el.meta["impactz_type"] = card.itype
        if ideal:
            self.rep.equivalent("IMPACTZ_IDEAL_CAVITY",
                                f"type {card.itype} with a negative file ID runs IMPACT-Z's "
                                "ideal-cavity model: gain = gradient·L·cos(phase)",
                                element=name, kind=el.kind)
            return self._common(el, card)
        fname = rfdata_name(file_id)
        if isinstance(el, FieldMap):
            el.files = [fname]
        path = self.dir / fname
        if path.is_file():
            el.meta["impactz_rfdata_path"] = str(path)
            if self.load_rfdata:
                rows = read_rfdata(path)
                el.meta["impactz_rfdata_columns"] = len(rows[0]) if rows else 0
                el.meta["impactz_rfdata"] = rows
        else:
            self.rep.lossy("IMPACTZ_RFDATA_MISSING",
                           f"{fname} (referenced by a type-{card.itype} element) is not "
                           "next to the deck; the field profile is unknown",
                           element=name, kind=el.kind, line=card.line, file=fname)
        self.rep.equivalent("IMPACTZ_RF_DRIVEN_PHASE",
                            f"type {card.itype} phase {card.v(3):g}° is a driven RF phase "
                            "(gain \u221d cos(2\u03c0f\u00b7t + \u03b80)), not a synchronous phase",
                            element=name, kind=el.kind)
        return self._common(el, card)

    # -- negative (BPM) type codes ---------------------------------------
    def _tm2(self, card: Card) -> Element:
        el = Instrument(name=self._name(card, "dump"), family="PHASE_DUMP",
                        params={"file": card.mapstp, "sample_period": card.v(1)})
        self.rep.exact(el.name, "Instrument", code="OK",
                       message="type -2 phase-space dump kept as an Instrument")
        return self._common(el, card, passthrough=True)

    def _tm13(self, card: Card) -> Element:
        el = Collimator(name=self._name(card, "slit"), length=card.length)
        x0, x1, y0, y1 = card.v(2), card.v(3), card.v(4), card.v(5)
        el.aperture = ApertureP(shape="RECTANGULAR", x_limits=(x0, x1), y_limits=(y0, y1))
        self.rep.exact(el.name, "Collimator", code="OK",
                       message="type -13 rectangular slit")
        return self._common(el, card)

    def _tm21(self, card: Card) -> Element:
        el = Kicker(name=self._name(card, "kick"), length=card.length,
                    hkick=card.v(3), vkick=card.v(5))
        extra = {"dx_m": card.v(2), "dy_m": card.v(4), "dz_deg": card.v(6),
                 "dPz_MeV": card.v(7)}
        if any(extra.values()):
            self.rep.lossy("IMPACTZ_CENTROID_SHIFT",
                           "type -21 also shifts x/y/phase/energy; only the angular "
                           "kicks map onto the IR Kicker",
                           element=el.name, kind="Kicker", line=card.line, **extra)
        else:
            self.rep.exact(el.name, "Kicker", code="OK",
                           message="type -21 centroid shift used as a thin kicker")
        return self._common(el, card)


def _brho(h: Header) -> float:
    """Signed rigidity of the header's reference particle [T·m]."""
    g = 1.0 + h.kinetic_energy_eV / h.mass_eV if h.mass_eV else 1.0
    pc = math.sqrt(max(g * g - 1.0, 0.0)) * h.mass_eV
    q = h.charge or 1.0
    return math.copysign(pc / (C_LIGHT * abs(q)), q)


def _header_meta(h: Header) -> dict:
    return {"npcol": h.npcol, "nprow": h.nprow, "dim": h.dim, "np": h.np,
            "flagmap": h.flagmap, "flagerr": h.flagerr, "flagdiag": h.flagdiag,
            "grid": [h.nx, h.ny, h.nz], "flagbc": h.flagbc,
            "xrad": h.xrad, "yrad": h.yrad, "perdlen": h.perdlen,
            "flagdist": h.flagdist, "rstartflg": h.rstartflg,
            "flagsbstp": h.flagsbstp, "nchrg": h.nchrg,
            "nptlist": list(h.nptlist), "currlist": list(h.currlist),
            "qmcclist": list(h.qmcclist), "distparam": list(h.distparam),
            "current_A": h.current_A, "charge": h.charge,
            "phase_ini_rad": h.phase_ini_rad}


def _bind_rf_types() -> None:
    for code in sorted(RF_TYPES):
        def make(c: int):
            def fn(self: Reader, card: Card) -> Element:
                return self._rf_element(card, kind=TYPE_NAMES[c])
            fn.__name__ = f"_t{c}"
            return fn
        setattr(Reader, f"_t{code}", make(code))


_bind_rf_types()
