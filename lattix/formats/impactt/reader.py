"""IMPACT-T ``ImpactT.in`` reader.

Everything below was read out of the IMPACT-T sources at
``particle_tracking_codes/tier1_linac/IMPACT-T`` (commit 75de6c3, 2025-10-03) and, where marked
MEASURED, confirmed by running the conda-forge ``impact-t`` 3.1.5 binary (``ImpactTexe``) on
2026-09-05 — see ``docs/formats/impactt.md`` and ``docs/oracles.md``.

File shape (``examples/Sample1/ImpactT.in``, ``src/Contrl/Input.f90``)
------------------------------------------------------------------------
A line whose first character is ``!`` is a comment; the nine header records are the first nine
non-comment lines:

===  =====================================================================================
 1   ``npcol nprow``                                  processor grid
 2   ``dt ntstep nbunch``                             time step [s], number of steps, bunches
 3   ``dim np flagmap flagerr flagdiag flagimg zimage``  6-D, particles, integrator flag
     (unused), misalignment flag, diagnostics flag, image-charge flag and cut-off [m]
 4   ``nx ny nz flagbc xrad yrad zleng``              space-charge mesh, boundary code,
     pipe half-widths [m], longitudinal domain [m]
 5   ``flagdist rstartflg flagsbstp nemission temission``  distribution type (16 = read
     ``partcl.data``), restart, sub-cycling, emission steps (≤ 0: none), emission time [s]
 6–8 ``distparam(1:21)``                              seven per plane
 9   ``current energy mass charge frequency phase``   current [A], reference KINETIC energy
     [eV], rest mass [eV], charge [e], scale frequency [Hz], initial phase [rad]
===  =====================================================================================

then one card per element, ``length nseg mapstp type zedge v1 … v23 /`` (``Fortran``
list-directed input stops at the ``/``; missing trailing columns are 0).  Elements are placed
by their absolute starting edge ``zedge`` (``Param(1)``), so drifts are not needed and cards
may overlap.  The reader turns the gaps between consecutive elements into ``Drift`` elements
marked ``native['impactt']['implicit']`` (the writer does not emit them again) and keeps every
raw column in ``native['impactt']`` so an IMPACT-T → IR → IMPACT-T round trip re-emits foreign
cards unchanged.

Type codes and their ``Param`` layout (``src/Appl/*.f90`` headers; ``Param(1)`` is ``zedge``):

* ``0`` drift: radius.  ``1`` quadrupole: gradient [T/m], file id (``0 < id < 100``: analytic
  fringe of that length, ``≥ 100``: profile file), radius, dx, dy, rot_x, rot_y, rot_z, RF
  frequency and phase of an RF quadrupole.  ``2`` constant focusing (no IR kind).
  ``3`` solenoid: ``Bz0`` [T] (the scale of the table), file id of the ``1T<id>.T7`` ``(r, z)``
  table (``Data.f90`` ``read2tsol_Data``; the tracked field ``getfldt_Sol`` has no analytic
  form and a missing table stops the run), radius — the table's radial extent, particles
  beyond it stop.  ``4`` dipole: ``Bx``, ``By`` [T], file id, radius, misalignments; the tracked field
  (``getfldt_Dipole``) uses ``By`` only and reads the pole faces from the ``rfdataN`` file
  (``utilities/chicaneImpt.f90``): ``csr_flag, γ_entrance, k1, b1, k2, b2, k3, b3, k4, b4``
  (four lines ``z = k·x + b`` relative to ``zedge``: entrance face, end of the entrance fringe,
  start of the exit fringe, exit face), ``z01, z02`` (Enge shifts), ``c1…c8`` (Enge
  coefficients), ``zcsr1, zcsr2``; the element length is the reference **arc** length.
  ``5`` multipole: id (2 sextupole, 3 octupole, 4 decapole), strength (``B''`` [T/m²] …), file
  id, radius, misalignments.  ``101/102/103`` DTL/CCDTL/CCL, ``104`` SC cavity, ``105`` SolRF,
  ``110–113`` EM field maps: ``scale``, RF frequency [Hz], ``theta0`` [deg], file id, radius,
  misalignments (105 adds ``Bz0``).  Negative types are run controls: ``-1`` centroid shift
  (``tsteer, dx, dpx, dy, dpy, dz, dpz``; ``dpx`` in γβ units — a thin kicker), ``-2``
  phase-space dump at ``Param(3)`` into ``fort.<mapstp>``, ``-4`` time-step change, ``-11``
  collimator (``tcol, xmin, xmax, ymin, ymax, flag`` — ≤ 10 rectangular, else round), ``-12``
  external linear map (``linearmap.in``), ``-99`` end.

RF (``getfldt_SC``, MEASURED): ``rfdataN`` for type 104 holds the Fourier coefficients of the
on-axis ``Ez`` only (one per line, ``a0, a1, b1, a2, b2, …``) with the **element length as the
period** and the element centre as the phase reference; the field is ``scale·Ez·cos(2πf·t + θ0)``
with ``t`` the absolute time from the start of the run, so ``θ0`` is a driven phase.  The reader
recovers the IR's synchronous phase and effective voltage by integrating that profile at the
reference velocity with the time of flight of lattix's walk (EQUIVALENT
``IMPACTT_RF_DRIVEN_PHASE``).  Type 105 files carry the ``N / zstart / zend / zlength``
header of each Fourier block (E, then B) and are kept as passthrough.

Coordinates (MEASURED, ``Output.f90`` ``phase_Output``): the phase-space dumps are
``x [m], γβ_x, y [m], γβ_y, z [m], γβ_z, q/m [e/eV], weight, id`` at one instant of time.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

from lattix.fidelity import FidelityReport
from lattix.formats.impactt.rfprofile import C_LIGHT, gain_from_profile, integrate_reference
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Element,
    Foil,
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
from lattix.ir.lattice import Lattice
from lattix.ir.reference import SPECIES, ReferenceParticle, Species
from lattix.ir.walk import energy_gain_eV

TYPE_NAMES: dict[int, str] = {
    0: "drift", 1: "quadrupole", 2: "constfoc", 3: "solenoid", 4: "dipole", 5: "multipole",
    101: "dtl", 102: "ccdtl", 103: "ccl", 104: "sc_cavity", 105: "solrf",
    110: "emfld", 111: "emfld_cart", 112: "emfld_cyl", 113: "emfld_analytic",
    -1: "steer", -2: "phase_dump", -3: "restart", -4: "dt_change", -5: "sc_3d_start",
    -6: "wakefield", -7: "merge", -8: "sc_change", -9: "slice_output", -11: "collimator",
    -12: "linear_map", -13: "dielectric_wake", -15: "point_to_point", -16: "heating",
    -17: "rotation", -18: "field_output", -99: "end",
}
#: run controls the reader keeps as ``Directive`` elements (zero length at their ``zedge``)
CONTROL_TYPES: frozenset[int] = frozenset({-3, -4, -5, -6, -7, -8, -9, -12, -13, -15, -16, -17, -18})
RF_TYPES: frozenset[int] = frozenset({101, 102, 103, 104, 105, 110, 111, 112, 113})
MULTIPOLE_ORDER: dict[int, int] = {2: 2, 3: 3, 4: 4}

_FORTRAN_EXP = re.compile(r"(?<=[\d.])[dD]([-+]?\d)")
_TAG = re.compile(r"lattix:\s*(.*)$")
_KV = re.compile(r"(\w+)=(\S+)")
_NAME_TAG = re.compile(r"^\s*!\s*lattice:\s*'([^']*)'")
_TOL = 1e-12


def _to_float(tok: str) -> float:
    return float(_FORTRAN_EXP.sub(r"e\1", tok))


def _is_comment(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("!")


def split_data_line(line: str) -> tuple[list[str], str]:
    """Fortran list-directed input: ``/`` ends the record, the rest is free text."""
    body, sep, rest = line.partition("/")
    return body.replace(",", " ").split(), (rest.strip() if sep else "")


@dataclass
class Card:
    """One element line: ``length nseg mapstp type zedge value2 … /``."""

    length: float
    nseg: int
    mapstp: int
    itype: int
    values: list[float] = field(default_factory=list)      # Param(1) = zedge, Param(2) = v1, …
    line: int = 0
    comment: str = ""
    tag: dict[str, str] = field(default_factory=dict)

    def v(self, i: int) -> float:
        """``Param(i)`` (1-based), 0.0 when the card does not supply it."""
        return self.values[i - 1] if 0 < i <= len(self.values) else 0.0

    @property
    def zedge(self) -> float:
        return self.v(1)

    @property
    def name(self) -> str:
        return TYPE_NAMES.get(self.itype, f"type{self.itype}")


@dataclass
class Header:
    """The nine header records of an ``ImpactT.in``."""

    npcol: int = 1
    nprow: int = 1
    dt_s: float = 1e-12
    ntstep: int = 1000000
    nbunch: int = 1
    dim: int = 6
    np: int = 1000
    flagmap: int = 1
    flagerr: int = 0
    flagdiag: int = 1
    flagimg: int = 0
    zimage: float = 0.016
    nx: int = 32
    ny: int = 32
    nz: int = 32
    flagbc: int = 1
    xrad: float = 1.0
    yrad: float = 1.0
    perdlen: float = 1.0e5
    flagdist: int = 2
    rstartflg: int = 0
    flagsbstp: int = 0
    nemission: int = -1
    temission: float = 1e-12
    distparam: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] * 3)
    current_A: float = 0.0
    kinetic_energy_eV: float = 0.0
    mass_eV: float = 0.0
    charge: float = 1.0
    frequency_Hz: float = 0.0
    phase_ini_rad: float = 0.0

    def to_meta(self) -> dict:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_meta(cls, d: dict) -> Header:
        h = cls()
        for k, v in (d or {}).items():
            if hasattr(h, k):
                setattr(h, k, list(v) if isinstance(v, list) else v)
        return h


def parse_deck(text: str) -> tuple[Header, list[Card], list[str]]:
    """Split an ``ImpactT.in`` into its header, its element cards and its comment lines."""
    records: list[tuple[int, str]] = []
    tags: dict[int, dict[str, str]] = {}
    comments: list[str] = []
    pending: dict[str, str] = {}
    for i, line in enumerate(text.splitlines(), start=1):
        if _is_comment(line):
            comments.append(line.rstrip())
            m = _TAG.search(line)
            if m:
                pending = dict(_KV.findall(m.group(1)))
            continue
        records.append((i, line))
        if pending:
            tags[len(records) - 1] = pending
            pending = {}
    if len(records) < 9:
        raise ValueError(f"ImpactT.in needs 9 header records, found {len(records)}")
    nums: list[list[float]] = []
    for _, line in records[:9]:
        toks, _ = split_data_line(line)
        nums.append([_to_float(t) for t in toks])

    def need(row: int, n: int) -> list[float]:
        if len(nums[row]) < n:
            raise ValueError(f"ImpactT.in header line {row + 1} needs {n} numbers, got {len(nums[row])}")
        return nums[row]

    h = Header()
    h.npcol, h.nprow = (int(v) for v in need(0, 2)[:2])
    r = need(1, 3)
    h.dt_s, h.ntstep, h.nbunch = r[0], int(r[1]), int(r[2])
    r = need(2, 5) + [0.0, 0.016]
    h.dim, h.np, h.flagmap, h.flagerr, h.flagdiag = (int(v) for v in r[:5])
    h.flagimg, h.zimage = int(r[5]), r[6]
    r = need(3, 7)
    h.nx, h.ny, h.nz, h.flagbc = (int(v) for v in r[:4])
    h.xrad, h.yrad, h.perdlen = r[4:7]
    r = need(4, 5)
    h.flagdist, h.rstartflg, h.flagsbstp, h.nemission = (int(v) for v in r[:4])
    h.temission = r[4]
    dist: list[float] = []
    for row in (5, 6, 7):
        dist.extend(need(row, 7)[:7])
    h.distparam = dist
    r = need(8, 6)
    (h.current_A, h.kinetic_energy_eV, h.mass_eV, h.charge, h.frequency_Hz, h.phase_ini_rad) = r[:6]
    cards: list[Card] = []
    for idx, (lineno, line) in enumerate(records[9:], start=9):
        toks, rest = split_data_line(line)
        if len(toks) < 4:
            raise ValueError(f"line {lineno}: an IMPACT-T element card needs at least "
                             f"'length nseg mapstp type', got {line.strip()!r}")
        vals = [_to_float(t) for t in toks]
        c = Card(length=vals[0], nseg=int(vals[1]), mapstp=int(vals[2]), itype=int(vals[3]),
                 values=vals[4:], line=lineno, comment=rest, tag=tags.get(idx, {}))
        cards.append(c)
        if c.itype == -99:
            break
    return h, cards, comments


def species_from_header(h: Header) -> Species:
    """A named species when the header's mass matches one to 1e-6, else a custom one."""
    q = int(round(h.charge)) or 1
    for spc in SPECIES.values():
        if spc.charge == q and h.mass_eV and abs(spc.mass_eV - h.mass_eV) <= 1e-6 * spc.mass_eV:
            return spc
    return Species(name=f"custom_m{h.mass_eV:.0f}_q{q:+d}", mass_eV=h.mass_eV, charge=q)


def rfdata_name(file_id: int) -> str:
    """``rfdataN`` (``Data.f90`` ``read1t_Data``: no suffix, up to three digits)."""
    return f"rfdata{int(file_id)}"


def solenoid_table_name(file_id: int) -> str:
    """``1TN.T7`` — the (r, z) table of a type-3 solenoid (``Data.f90`` ``read2tsol_Data``)."""
    return f"1T{int(file_id)}.T7"


def read_rfdata(path: Path) -> list[float]:
    """The first number of every line (``read1t_Data`` reads one value per record)."""
    out: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        toks, _ = split_data_line(line)
        if not toks or line.lstrip().startswith("!"):
            continue
        try:
            out.append(_to_float(toks[0]))
        except ValueError:
            continue
    return out


def card_span(card: Card) -> tuple[float, float]:
    """``(entrance, length)`` of a card as an element: a lattix ``pad=…`` tag (a solenoid whose
    field table ramps beyond the hard-edge ends) restores the nominal span."""
    tag = card.tag
    if "pad" in tag and "L" in tag:
        try:
            return round(card.zedge + float(tag["pad"]), 12), float(tag["L"])
        except ValueError:
            pass
    return card.zedge, card.length


def solenoid_table_integrals(rows: list[list[float]]) -> tuple[float, float, float] | None:
    """``(∫Bz dz, ∫Bz² dz, table length)`` of a normalised type-3 table (``read2tsol_Data``: two
    header lines ``Rmin Rmax NrIntv`` and ``Zmin Zmax NzIntv`` in cm, then ``Br Bz`` rows with r
    fastest), on the axis."""
    if len(rows) < 3 or len(rows[0]) < 3 or len(rows[1]) < 3:
        return None
    nr, nz = int(rows[0][2]), int(rows[1][2])
    zlen = (rows[1][1] - rows[1][0]) / 100.0
    if nz < 1 or zlen <= 0:
        return None
    bz = [rows[2 + j * (nr + 1)][1] for j in range(nz + 1) if 2 + j * (nr + 1) < len(rows)]
    if len(bz) < 2:
        return None
    hz = zlen / nz
    i1 = sum(0.5 * hz * (bz[j] + bz[j + 1]) for j in range(len(bz) - 1))
    i2 = sum(hz * (bz[j] ** 2 + bz[j] * bz[j + 1] + bz[j + 1] ** 2) / 3.0 for j in range(len(bz) - 1))
    return i1, i2, zlen


def _merge_thin_multipoles(elements: list[Element]) -> list[Element]:
    """The writer emits one short card per order of a thin multipole: fold consecutive restored
    orders of the same element back into one ``Multipole``."""
    out: list[Element] = []
    for e in elements:
        tag = e.native.get("impactt", {}).get("thin_of") if e.kind == "Multipole" else None
        prev = out[-1] if out else None
        if (tag and prev is not None and prev.kind == "Multipole"
                and prev.native.get("impactt", {}).get("thin_of") == tag and prev.length == 0):
            for attr in ("BnL", "BsL"):
                d = dict(getattr(prev.multipole, attr))
                for k, v in getattr(e.multipole, attr).items():
                    d[k] = d.get(k, 0.0) + v
                setattr(prev.multipole, attr, d)
            continue
        out.append(e)
    return out


def _lattice_name(comments) -> str | None:
    for c in comments or ():
        m = _NAME_TAG.match(str(c))
        if m:
            return m.group(1)
    return None


def _brho_signed(ref: ReferenceParticle) -> float:
    return ref.brho_signed


class Reader:
    """``Reader().read(path)`` → ``(Lattice, FidelityReport)``."""

    format = "impactt"

    PROMOTIONS: dict[str, type[Element]] = {
        "Marker": Marker, "Instrument": Instrument, "Foil": Foil, "Taylor": Taylor,
        "Patch": Patch, "RFQCell": RFQCell, "Superposition": Superposition,
        "ReferenceChange": ReferenceChange, "NCells": NCells,
    }

    def read(self, path: Path, *, strict: bool = False, encoding: str = "utf-8",
             load_rfdata: bool = True, **_options) -> tuple[Lattice, FidelityReport]:
        path = Path(path)
        rep = FidelityReport(source_format="impactt", source_file=str(path))
        header, cards, comments = parse_deck(path.read_text(encoding=encoding, errors="replace"))
        sp = species_from_header(header)
        ref0 = ReferenceParticle(species=sp, kinetic_energy_eV=header.kinetic_energy_eV,
                                 rf_frequency_Hz=header.frequency_Hz or None)
        # the deck's time origin is the reference at z = 0: the walk's clock starts at the nominal
        # start with the field-free flight up to it
        first = min((card_span(c)[0] for c in cards if c.itype != -99), default=0.0)
        ref0 = ref0.model_copy(update={"time_s": first / (ref0.beta * C_LIGHT)})
        self.rep = rep
        self.header = header
        self.dir = path.parent
        self.load_rfdata = load_rfdata
        self._counts: dict[str, int] = {}
        self._names: set[str] = set()
        self._t_corr = 0.0                    # IMPACT-T's clock minus the IR walk's (see the writer)
        elements = _merge_thin_multipoles(self._assemble(cards, ref0))
        lat = Lattice.from_sequence(_lattice_name(comments) or path.stem or "impactt", elements, ref0)
        lat.meta.update({"source_format": "impactt", "impactt_header": header.to_meta(), "impactt_s_offset": first})
        rep.equivalent("SIM_SETTINGS_KEPT_IN_META",
                       "the run settings (time step, particles, mesh, distribution) are kept in "
                       "meta['impactt_header'], not in the IR")
        if header.current_A:
            rep.lossy("IMPACTT_BEAM_CURRENT",
                      f"deck current {header.current_A:g} A (space charge) is not part of the IR",
                      details={"current_A": header.current_A})
        rep.raise_if(strict)
        return lat, rep

    # ------------------------------------------------------------------ layout
    def _assemble(self, cards: list[Card], ref0: ReferenceParticle) -> list[Element]:
        """Order the cards by ``zedge``, fill the gaps with implicit drifts, split thick elements
        around zero-length cards inside them and convert each card with the running reference."""
        physical = [c for c in cards if c.itype != -99]
        spans = [(round(a, 12), b) for a, b in (card_span(c) for c in physical)]   # one picometre grid for all
        order = sorted(range(len(physical)), key=lambda i: (spans[i][0], i))
        thick = [(spans[i][0], spans[i][0] + spans[i][1], i) for i in order if spans[i][1] > _TOL]
        inside: dict[int, list[int]] = {i: [] for _, _, i in thick}
        units: list[tuple[float, int, int, list[int]]] = []      # (entrance, rank, card index, inner)
        for i in order:
            if spans[i][1] > _TOL:
                continue
            pos = spans[i][0]                     # a thin element written with a length: its nominal position
            host = next((h for a, b, h in thick if a + _TOL < pos < b - _TOL), None)
            if host is None:
                units.append((pos, 0, i, []))
            else:
                inside[host].append(i)
        for a, _, i in thick:
            units.append((a, 1, i, sorted(inside[i], key=lambda j: physical[j].zedge)))
        units.sort(key=lambda u: (u[0], u[1]))
        elements: list[Element] = []
        ref = ref0
        s = 0.0
        n_gap = 0
        first = True
        prev_surrogate = False
        surrogate_at = {round(spans[i][0], 12) for i in range(len(physical))
                        if physical[i].tag.get("L") == "0" and "pad" in physical[i].tag}
        for entrance, _, i, inner in units:
            c = physical[i]
            c_length = spans[i][1]
            # a thin element the writer gave a length (``L=0 pad=…``): the drifts around it gave that
            # length up, and get it back here instead of an implicit gap
            surrogate = c.tag.get("L") == "0" and "pad" in c.tag
            if first:
                s = entrance                      # the first card sets the origin (Sample1 starts below 0)
                first = False
            gap = entrance - s
            extend = 0.0
            if gap > 1e-9:
                if (surrogate or round(entrance, 12) in surrogate_at) and elements and elements[-1].kind == "Drift":
                    elements[-1].length = round(float(elements[-1].length) + gap, 12)
                    ref = ref.advanced(ds_m=gap)
                elif prev_surrogate and c.itype == 0 and c_length > _TOL and c.tag.get("kind", "Drift") == "Drift":
                    extend = gap
                else:
                    n_gap += 1
                    d = Drift(name=self._unique(f"gap_{n_gap}"), length=gap)
                    d.native["impactt"] = {"implicit": True}
                    elements.append(d)
                    ref = ref.advanced(ds_m=gap)
                s = entrance
            prev_surrogate = surrogate or (prev_surrogate and c_length <= _TOL)
            pushed = gap < -1e-9
            if pushed:
                entrance = s                      # the IR is a sequence: the card follows the previous element
                self.rep.lossy("IMPACTT_OVERLAP",
                               f"card at line {c.line} (type {c.itype}) starts {-gap:.3g} m before the "
                               "previous element ends; the IR places it at the previous element's exit "
                               "(a rewrite keeps the card's own zedge)",
                               kind=c.name, line=c.line, overlap_m=-gap)
            el = self._element(c, ref)
            if el is None:
                continue
            if extend:
                el.length = round(float(el.length) + extend, 12)
                entrance -= extend
                c_length += extend
            if pushed and "impactt" in el.native:
                el.native["impactt"]["pushed"] = True
            folded = self._fold_thick(c, el, inner, physical, ref)
            if folded is not None:
                elements.append(folded)
                ref = ref.advanced(ds_m=folded.length)
                s = entrance + c_length
                continue
            if inner and c.itype in RF_TYPES:
                # an RF card is one field and is never split: zero-length cards inside it go to its
                # entrance or its exit, whichever is nearer
                mid = entrance + 0.5 * c_length
                for where, group in (("entrance", [j for j in inner if physical[j].zedge <= mid]),
                                     ("exit", [j for j in inner if physical[j].zedge > mid])):
                    if where == "exit":
                        elements.append(el)
                        ref = ref.advanced(dE_eV=energy_gain_eV(el, ref, None), ds_m=el.length)
                    for j in group:
                        thin = self._element(physical[j], ref)
                        if thin is not None:
                            self.rep.equivalent("IMPACTT_THIN_INSIDE_THICK",
                                                f"zero-length card {thin.name!r} sits inside the RF card {el.name!r}: "
                                                f"placed at its {where} (RF cards are not split)",
                                                element=thin.name, kind=thin.kind)
                            elements.append(thin)
                            ref = ref.advanced(dE_eV=energy_gain_eV(thin, ref, None))
                s = entrance + c_length
                continue
            if inner:
                cursor = entrance
                n_part = 0
                for j in inner:
                    cj = physical[j]
                    seg = cj.zedge - cursor
                    if seg > 1e-9:
                        n_part += 1
                        part = el.model_copy(update={"name": f"{el.name}_{n_part}", "length": seg})
                        elements.append(part)
                        ref = ref.advanced(dE_eV=energy_gain_eV(part, ref, None), ds_m=seg)
                        cursor = cj.zedge
                    thin = self._element(cj, ref)
                    if thin is not None:
                        if el.kind != "Drift":
                            self.rep.equivalent("IMPACTT_THIN_INSIDE_THICK",
                                                f"zero-length card {thin.name!r} sits inside {el.name!r}: "
                                                "the element is split around it",
                                                element=thin.name, kind=thin.kind)
                        elements.append(thin)
                        ref = ref.advanced(dE_eV=energy_gain_eV(thin, ref, None))
                seg = entrance + c_length - cursor
                if seg > 1e-9:
                    n_part += 1
                    part = el.model_copy(update={"name": f"{el.name}_{n_part}", "length": seg})
                    elements.append(part)
                    ref = ref.advanced(dE_eV=energy_gain_eV(part, ref, None), ds_m=seg)
            else:
                elements.append(el)
                ref = ref.advanced(dE_eV=energy_gain_eV(el, ref, None), ds_m=el.length)
            s = entrance + c_length
        return elements

    def _fold_thick(self, card: Card, el: Element, inner: list[int], physical: list[Card],
                    ref: ReferenceParticle) -> Element | None:
        """A drift card tagged ``kind=Kicker``/``Collimator`` holding exactly one ``-1``/``-11`` card of
        the same name is the writer's picture of a thick kicker or collimator: one element again."""
        want = card.tag.get("kind")
        if el.kind != "Drift" or want not in ("Kicker", "Collimator") or len(inner) != 1:
            return None
        cj = physical[inner[0]]
        if cj.tag.get("name") != card.tag.get("name") or cj.itype != {"Kicker": -1, "Collimator": -11}[want]:
            return None
        thin = self._element(cj, ref)
        if thin is None or thin.kind != want:
            return None
        self._names.discard(thin.name)
        out = thin.model_copy(update={"name": el.name, "length": el.length, "aperture": thin.aperture or el.aperture})
        out.native["impactt"] = dict(el.native.get("impactt", {}))
        self.rep.exact(out.name, want, code="IMPACTT_NAME_TAG",
                       message=f"restored a thick {want} from its drift card and the -{-cj.itype} inside it")
        return out

    # ------------------------------------------------------------------ helpers
    def _unique(self, name: str) -> str:
        base, n = name, 2
        while name in self._names:
            name = f"{base}_{n}"
            n += 1
        self._names.add(name)
        return name

    def _name(self, card: Card, stem: str) -> str:
        if "name" in card.tag:
            return self._unique(unquote(card.tag["name"]))
        self._counts[stem] = self._counts.get(stem, 0) + 1
        return self._unique(f"{stem}_{self._counts[stem]}")

    @staticmethod
    def _aperture(radius: float) -> ApertureP | None:
        return ApertureP.circle(radius) if radius > 0 else None

    @staticmethod
    def _shift(card: Card, first: int) -> BodyShiftP | None:
        """``dx, dy, rot_x, rot_y, rot_z`` starting at ``Param(first)``."""
        vals = [card.v(first + k) for k in range(5)]
        if not any(vals):
            return None
        return BodyShiftP(x_offset=vals[0], y_offset=vals[1], x_rot=vals[2], y_rot=vals[3], tilt=vals[4])

    def _native(self, card: Card, *, passthrough: bool = False) -> dict:
        d = {"type": card.itype, "class": card.name, "nseg": card.nseg, "mapstp": card.mapstp,
             "values": list(card.values), "line": card.line}
        if card.comment:
            d["comment"] = card.comment
        if passthrough:
            d["passthrough"] = True
        return d

    def _common(self, el: Element, card: Card, *, passthrough: bool = False) -> Element:
        el.native["impactt"] = self._native(card, passthrough=passthrough)
        el.tracking.setdefault("nseg", card.nseg)
        el.tracking.setdefault("mapstp", card.mapstp)
        el.provenance = Provenance(format="impactt", line=card.line, original_type=card.name)
        return self._promote(el, card)

    def _promote(self, el: Element, card: Card) -> Element:
        """Rebuild a lattix-written element that had to be degraded to a drift."""
        want = card.tag.get("kind")
        cls = self.PROMOTIONS.get(want or "")
        if cls is None or el.kind != "Drift" or want == el.kind:
            return el
        out = cls(name=el.name, length=el.length, aperture=el.aperture, native=dict(el.native),
                  tracking=dict(el.tracking), provenance=el.provenance, meta=dict(el.meta))
        self.rep.exact(el.name, want, code="IMPACTT_NAME_TAG",
                       message=f"restored from the '! lattix:' tag ({want})")
        return out

    def _rfdata(self, card: Card, file_id: int, el: Element) -> list[float] | None:
        fname = rfdata_name(file_id)
        path = self.dir / fname
        el.meta["impactt_file_id"] = file_id
        if not path.is_file():
            self.rep.lossy("IMPACTT_RFDATA_MISSING",
                           f"{fname} (referenced by a type-{card.itype} card) is not next to the deck",
                           element=el.name, kind=el.kind, line=card.line, file=fname)
            return None
        el.meta["impactt_rfdata_path"] = str(path)
        if not self.load_rfdata:
            return None
        coefs = read_rfdata(path)
        el.meta["impactt_rfdata"] = coefs
        return coefs

    # ------------------------------------------------------------------ dispatch
    def _element(self, card: Card, ref: ReferenceParticle) -> Element | None:
        t = card.itype
        if t in CONTROL_TYPES:
            return self._control(card)
        fn = getattr(self, f"_t{t}" if t >= 0 else f"_tm{-t}", None)
        if fn is None:
            return self._unsupported(card)
        return fn(card, ref)

    def _unsupported(self, card: Card) -> Element:
        name = self._name(card, TYPE_NAMES.get(card.itype, "type"))
        el: Element = Drift(name=name, length=card.length) if card.length > _TOL else Marker(name=name)
        self.rep.dropped("UNSUPPORTED_IMPACTT_ELEMENT",
                         f"IMPACT-T type {card.itype} ({card.name}) has no IR kind; kept as a {el.kind} "
                         "with the raw columns in native['impactt'] (re-emitted unchanged)",
                         element=name, kind=el.kind, line=card.line)
        return self._common(el, card, passthrough=True)

    def _control(self, card: Card) -> Element:
        el = Directive(name=self._name(card, card.name), format="impactt", card=str(card.itype),
                       args=[repr(v) for v in card.values], role="other")
        self.rep.exact(el.name, "Directive", code="IMPACTT_RUN_CONTROL",
                       message=f"type {card.itype} ({card.name}) run control kept as a Directive")
        return self._common(el, card, passthrough=True)

    def _t0(self, card: Card, ref) -> Element:
        el = Drift(name=self._name(card, "drift"), length=card.length, aperture=self._aperture(card.v(2)))
        return self._common(el, card)

    def _thin_multipole(self, card: Card, order: int, strength: float, rot_col: int, radius: float) -> Element:
        """A thin multipole the writer gave a length (tag ``L=0 pad=…``): one order per card, the
        skew part as a roll of the normal strength; the cards of one element are merged afterwards."""
        rot = card.v(rot_col)
        bnl, bsl = strength * card.length * math.cos(order * rot), strength * card.length * math.sin(order * rot)
        el = Multipole(name=self._name(card, "mult"), length=0.0, aperture=self._aperture(radius))
        el.multipole = MagneticMultipoleP(BnL={order: bnl} if abs(bnl) > 1e-300 or not bsl else {},
                                          BsL={order: bsl} if abs(bsl) > 1e-300 else {})
        self.rep.exact(el.name, "Multipole", code="IMPACTT_NAME_TAG",
                       message="thin multipole restored from the lattix tag (written as a short thick card)")
        out = self._common(el, card)
        out.native.setdefault("impactt", {})["thin_of"] = card.tag.get("name") or el.name
        return out

    def _t1(self, card: Card, ref) -> Element:
        if card.tag.get("L") == "0" and "pad" in card.tag:
            return self._thin_multipole(card, 1, card.v(2), 9, card.v(4))
        el = Quadrupole(name=self._name(card, "quad"), length=card.length, aperture=self._aperture(card.v(4)),
                        shift=self._shift(card, 5))
        el.multipole = MagneticMultipoleP(Bn={1: card.v(2)})
        fid = card.v(3)
        if 0.0 < fid < 100.0:
            self.rep.lossy("IMPACTT_QUAD_FRINGE_DROPPED",
                           f"quadrupole uses IMPACT-T's analytic fringe (file id {fid:g}); the IR keeps the "
                           "hard-edge gradient only", element=el.name, kind="Quadrupole", line=card.line)
        elif fid >= 100.0:
            self.rep.lossy("IMPACTT_QUAD_PROFILE",
                           f"quadrupole reads its gradient profile from {rfdata_name(int(fid))}; the IR keeps "
                           "the hard-edge gradient only", element=el.name, kind="Quadrupole", line=card.line)
        if card.v(10) or card.v(11):
            self.rep.lossy("IMPACTT_RF_QUADRUPOLE", "the RF-quadrupole frequency/phase columns have no IR field",
                           element=el.name, kind="Quadrupole", line=card.line)
        return self._common(el, card)

    def _t2(self, card: Card, ref) -> Element:
        return self._unsupported(card)

    def _t3(self, card: Card, ref) -> Element:
        """Type 3: ``Bz0``, file id, radius.  The tracker (``getfldt_Sol``) takes the field from a
        2-D ``(r, z)`` table in ``rfdataN`` scaled by ``Bz0``; lattix's own tables carry the nominal
        hard-edge solenoid in the card's tag."""
        entrance, length = card_span(card)
        tag = card.tag
        el = Solenoid(name=self._name(card, "sol"), length=length, shift=self._shift(card, 5),
                      solenoid=SolenoidP(Bsol_T=card.v(2)), aperture=self._aperture(card.v(4)))
        fid = int(round(card.v(3)))
        if "Bsol" in tag and "pad" in tag:
            el.solenoid.Bsol_T = float(tag["Bsol"])
            el.meta["impactt_table"] = {"Bz0": card.v(2), "length": card.length, "pad": float(tag["pad"]),
                                        "file_id": fid}
            path = self.dir / solenoid_table_name(fid)
            if fid > 0 and path.is_file():
                el.meta["impactt_rfdata_rows"] = [[float(t) for t in ln.replace(",", " ").split()[:3]]
                                                  for ln in path.read_text().splitlines()
                                                  if ln.strip() and not ln.lstrip().startswith("!")]
            self.rep.exact(el.name, "Solenoid", code="IMPACTT_NAME_TAG",
                           message="hard-edge solenoid restored from the lattix tag (the table ramps beyond its ends)")
            return self._common(el, card)
        if fid <= 0:
            self.rep.lossy("IMPACTT_SOLENOID_NO_TABLE",
                           "a type-3 solenoid without a field table cannot be tracked by IMPACT-T (Data.f90 "
                           "read2tsol_Data); the IR keeps Bz0 as a hard-edge field",
                           element=el.name, kind="Solenoid", line=card.line)
            return self._common(el, card)
        path = self.dir / solenoid_table_name(fid)
        el.meta["impactt_file_id"] = fid
        if not path.is_file():
            self.rep.lossy("IMPACTT_RFDATA_MISSING", f"{path.name} (the solenoid's field table) is not next to the "
                                                    "deck; Bz0 kept as a hard-edge field",
                           element=el.name, kind="Solenoid", line=card.line, file=path.name)
            return self._common(el, card)
        rows = [[float(t) for t in ln.replace(",", " ").split()[:3]] for ln in path.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("!")]
        ints = solenoid_table_integrals(rows)
        if ints is None:
            self.rep.lossy("IMPACTT_SOLENOID_TABLE_UNREAD", f"{path.name} is not a (r, z) solenoid table",
                           element=el.name, kind="Solenoid", line=card.line)
            return self._common(el, card)
        i1, i2, zlen = ints
        b0 = card.v(2)
        if i1 and i2:
            l_eff = i1 * i1 / i2
            el.solenoid.Bsol_T = b0 * i2 / i1
            el.length = min(l_eff, length) if length > 0 else l_eff
            self.rep.equivalent("IMPACTT_SOLENOID_TABLE",
                                f"the (r, z) field table becomes a hard-edge solenoid preserving ∫Bz and ∫Bz² "
                                f"(L_eff = {l_eff:.6g} m, B_eff = {el.solenoid.Bsol_T:.6g} T)",
                                element=el.name, kind="Solenoid", L_eff_m=l_eff, B_eff_T=el.solenoid.Bsol_T)
        return self._common(el, card)

    def _t4(self, card: Card, ref) -> Element:
        el = Bend(name=self._name(card, "bend"), length=card.length, aperture=self._aperture(card.v(5)),
                  shift=self._shift(card, 6))
        by, bx = card.v(3), card.v(2)
        brho = _brho_signed(ref)
        angle = card.length * by / brho if brho else 0.0
        el.bend = BendP(angle=angle)
        if bx:
            self.rep.lossy("IMPACTT_BEND_BX_DROPPED",
                           "the tracked dipole field (getfldt_Dipole) uses By only; Bx is dropped",
                           element=el.name, kind="Bend", line=card.line, Bx=bx)
        coefs = self._rfdata(card, int(round(card.v(4))), el) if card.v(4) > 0 else None
        if coefs and len(coefs) >= 10 and angle:
            s = 1.0 if angle > 0 else -1.0
            k1, k4 = coefs[2], coefs[8]
            el.bend.e1 = math.atan(s * k1)
            el.bend.e2 = abs(angle) - math.atan(s * k4)
            el.meta["impactt_bend"] = {"angle": angle, "e1": el.bend.e1, "e2": el.bend.e2}
            fringe = max(coefs[5] - coefs[3], coefs[9] - coefs[7]) if len(coefs) >= 10 else 0.0
            if fringe > 1e-6 or any(coefs[12:20]):
                self.rep.lossy("IMPACTT_BEND_FRINGE_MODEL",
                               "the Enge fringe of the pole-face file has no IR field (fint/hgap)",
                               element=el.name, kind="Bend", line=card.line, fringe_m=fringe)
        elif card.v(4) <= 0:
            self.rep.lossy("IMPACTT_BEND_NO_FILE",
                           "a type-4 dipole without a pole-face file cannot be tracked by IMPACT-T; "
                           "sector faces assumed", element=el.name, kind="Bend", line=card.line)
        return self._common(el, card)

    def _t5(self, card: Card, ref) -> Element:
        order = MULTIPOLE_ORDER.get(int(round(card.v(2))))
        if order is None:
            return self._unsupported(card)
        if card.tag.get("L") == "0" and "pad" in card.tag:
            return self._thin_multipole(card, order, card.v(3), 10, card.v(5))
        name = self._name(card, "mult")
        if order == 4:
            el: Element = Multipole(name=name, length=card.length, aperture=self._aperture(card.v(5)),
                                    shift=self._shift(card, 6))
            el.multipole = MagneticMultipoleP(BnL={4: card.v(3) * card.length})
            self.rep.equivalent("IMPACTT_DECAPOLE_AS_THIN",
                                "the IR has no thick decapole; kept as a thin multipole of the integrated "
                                "strength over the element's length", element=name, kind="Multipole")
        else:
            cls = {2: Sextupole, 3: Octupole}[order]
            el = cls(name=name, length=card.length, aperture=self._aperture(card.v(5)), shift=self._shift(card, 6))
            el.multipole = MagneticMultipoleP(Bn={order: card.v(3)})
        if card.v(4) > 1e-5:
            self.rep.lossy("IMPACTT_MULTIPOLE_PROFILE",
                           f"multipole reads its profile from {rfdata_name(int(card.v(4)))}",
                           element=name, kind=el.kind, line=card.line)
        return self._common(el, card)

    def _rf_element(self, card: Card, ref: ReferenceParticle) -> Element:
        name = self._name(card, card.name)
        file_id = int(round(card.v(5)))
        freq = card.v(3) or None
        thin = card.tag.get("L") == "0" and "pad" in card.tag          # a thin gap written as a short cavity
        pad = float(card.tag["pad"]) if thin else 0.0
        rf = RFP(frequency_Hz=freq, phase_rad=math.radians(card.v(4)), phase_is_sync=False,
                 L_active_m=None if thin else card.length)
        el: Element
        if card.itype in (101, 102, 103):
            el = NCells(name=name, length=card.length, rf=rf, aperture=self._aperture(card.v(6)),
                        shift=self._shift(card, 11 if card.itype == 101 else 7))
            if card.itype == 101:
                el.params = {"quad1_length_m": card.v(7), "quad1_gradient_T_per_m": card.v(8),
                             "quad2_length_m": card.v(9), "quad2_gradient_T_per_m": card.v(10)}
        else:
            el = RFCavity(name=name, length=0.0 if thin else card.length, rf=rf, aperture=self._aperture(card.v(6)),
                          shift=self._shift(card, 7))
            if thin:
                el.meta["impactt_thin"] = {"length": card.length, "pad": pad}
        el.meta["impactt_scale"] = card.v(2)
        el.meta["impactt_type"] = card.itype
        el.meta["impactt_theta0_deg"] = card.v(4)
        if card.itype == 105:
            el.meta["impactt_Bz0_T"] = card.v(12)
        coefs = self._rfdata(card, file_id, el) if file_id > 0 else None
        passthrough = card.itype != 104
        if card.itype == 104 and coefs and freq and card.length > 0:
            t_in = ref.time_s + self._t_corr - pad / (ref.beta * C_LIGHT)   # the card starts ``pad`` before the gap
            dE, v, psi = gain_from_profile(coefs, card.length, card.v(2), card.v(4), freq, t_in,
                                           ref.kinetic_energy_eV, ref.species.mass_eV, float(ref.species.charge))
            _, t_out = integrate_reference(coefs, card.length, card.v(2), card.v(4), freq, t_in,
                                           ref.kinetic_energy_eV, ref.species.mass_eV, float(ref.species.charge))
            ref_out = ref.advanced(dE_eV=dE)
            if thin:
                t_ir_exit = ref.time_s + self._t_corr + (card.length - pad) / (ref_out.beta * C_LIGHT)
            else:
                t_ir_exit = t_in + card.length / (ref.beta * C_LIGHT)
            self._t_corr += t_out - t_ir_exit
            rf.voltage_V, rf.phase_rad, rf.phase_is_sync, rf.dE_ref_eV = v, psi, True, dE
            el.meta["impactt_rf"] = {"scale": card.v(2), "theta0_deg": card.v(4), "voltage_V": v, "phase_rad": psi}
            self.rep.equivalent("IMPACTT_RF_DRIVEN_PHASE",
                                f"type 104 theta0 = {card.v(4):g}° is a driven phase (field ∝ cos(2πf·t + θ0)); "
                                f"V = {v:.6g} V is the largest gain over θ0 and the synchronous phase "
                                f"{math.degrees(psi):.3f}° follows from the reference integrated through the "
                                "profile at lattix's time of flight",
                                element=name, kind=el.kind)
        else:
            self.rep.lossy("IMPACTT_RF_GAIN_UNKNOWN",
                           f"type {card.itype} keeps its columns and rfdata as passthrough; the IR carries no "
                           "voltage for it", element=name, kind=el.kind, line=card.line)
            passthrough = True
            if card.itype == 105 and card.v(12):
                self.rep.lossy("IMPACTT_SOLRF_BZ_DROPPED",
                               f"the SolRF's solenoid field Bz0 = {card.v(12):g} T is not part of the IR cavity",
                               element=name, kind=el.kind, line=card.line)
        return self._common(el, card, passthrough=passthrough)

    def _tm1(self, card: Card, ref) -> Element:
        bg = ref.beta * ref.gamma
        el = Kicker(name=self._name(card, "kick"), length=0.0, hkick=card.v(4) / bg if bg else 0.0,
                    vkick=card.v(6) / bg if bg else 0.0)
        extra = {"dx_m": card.v(3), "dy_m": card.v(5), "dz_m": card.v(7), "dpz": card.v(8)}
        if any(extra.values()):
            self.rep.lossy("IMPACTT_CENTROID_SHIFT",
                           "type -1 also shifts x/y/z/pz; only the angular kicks map onto the IR Kicker",
                           element=el.name, kind="Kicker", line=card.line, **extra)
        else:
            self.rep.exact(el.name, "Kicker", code="OK", message="type -1 centroid momentum shift used as a "
                                                                 "thin kicker (dpx = hkick·γβ)")
        return self._common(el, card)

    def _tm2(self, card: Card, ref) -> Element:
        el = Instrument(name=self._name(card, "dump"), family="PHASE_DUMP",
                        params={"file": card.mapstp, "sample_period": card.nseg, "z_m": card.v(3)})
        self.rep.exact(el.name, "Instrument", code="OK", message="type -2 phase-space dump kept as an Instrument")
        return self._common(el, card, passthrough=True)

    def _tm11(self, card: Card, ref) -> Element:
        el = Collimator(name=self._name(card, "collimator"), length=0.0)
        shape = "RECTANGULAR" if card.v(7) <= 10.0 else "ELLIPTICAL"
        el.aperture = ApertureP(shape=shape, x_limits=(card.v(3), card.v(4)), y_limits=(card.v(5), card.v(6)))
        self.rep.exact(el.name, "Collimator", code="OK",
                       message=f"type -11 collimator ({shape.lower()}: flag {card.v(7):g})")
        return self._common(el, card)


def _bind_rf_types() -> None:
    for code in sorted(RF_TYPES):
        def make(c: int):
            def fn(self: Reader, card: Card, ref) -> Element:
                return self._rf_element(card, ref)
            fn.__name__ = f"_t{c}"
            return fn
        setattr(Reader, f"_t{code}", make(code))


_bind_rf_types()


def read(path: Path, **options) -> tuple[Lattice, FidelityReport]:
    return Reader().read(Path(path), **options)
