"""IMPACT-T ``ImpactT.in`` writer (see :mod:`lattix.formats.impactt.reader` for the file rules
and their evidence in the sources).

Positions: every card carries its absolute starting edge ``zedge`` (``Param(1)``), taken from
the IR's ``s`` of the placed element.  IMPACT-T re-bases its frame at a dipole exit so that the
following ``zedge`` continues from ``zedge + L`` with ``L`` the reference **arc** length — the
same bookkeeping as the IR's ``s`` (MEASURED 2026-09-05).  Drifts are written as type-0 cards
(a foreign deck's implicit gaps, marked ``native['impactt']['implicit']``, are not).

Conventions the writer encodes (all MEASURED with ``ImpactTexe`` 3.1.5 unless noted):

* quadrupole ``Param(2)`` is the lab gradient [T/m] and ``rot_z`` (``Param(9)``) a roll, so a
  skew term is written as a rotated normal one; file id 0 is a hard edge;
* solenoid ``Param(4)`` (the "radius") is the fringe length — ``Sol.f90`` ramps ``Bz`` over
  ``2·radius`` at each end — so a hard-edge solenoid is written with radius 0 and the aperture,
  which IMPACT-T does not use for losses, is recorded as dropped;
* dipole: ``By = Bρ_signed·θ/L_arc`` (``By > 0`` bends a positive particle towards ``−x``,
  MAD's positive angle: ``R16 > 0``); the card length is the arc and a generated ``rfdataN``
  holds the entrance γ and the pole faces as lines ``z = k·x + b`` in the entrance frame
  (``k = ±tan(e1)`` through the origin, ``k = ±tan(|θ| − e2)`` through the arc's end point),
  with 1 nm fringe zones (hard edges).  IMPACT-T tracks the whole bunch in the reference's
  rotating frame and applies the field for the reference's transit only, so its dipole map has
  no body or edge focusing for off-axis particles (``docs/oracles.md``): the translation is
  exact, the engine's dipole model is not;
* cavity: type 104 with a generated on-axis profile (a raised cosine over the active length,
  Fourier coefficients with the element length as the period), ``scale`` calibrated to the
  reference transit-time factor and ``theta0`` derived from lattix's time of flight so that the
  reference gains ``q·V·cos φs``; a thin gap becomes a short cavity centred at the gap whose
  length is taken from the neighbouring drifts (``THIN_GAP_AS_SHORT_CAVITY``);
* kicker: a type ``-1`` centroid momentum shift at the kicker's centre, ``dpx = hkick·γβ``
  (a thick kicker keeps its length as a drift, ``THICK_KICKER_SPLIT``);
* collimator: type ``-11`` (flag 1 rectangular, 11 round).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from lattix import __version__
from lattix.fidelity import FidelityReport
from lattix.formats.base import check_rules_coverage, with_rf_focusing
from lattix.formats.impactt.reader import Header, rfdata_name, solenoid_table_name
from lattix.formats.impactt.rfprofile import (
    C_LIGHT,
    calibrate,
    cell_train,
    fourier_coefficients,
    integrate_reference,
    raised_cosine,
)
from lattix.ir.elements import (
    Bend,
    Collimator,
    Directive,
    Element,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    Marker,
    Multipole,
    NCells,
    Patch,
    RFCavity,
    RFQCell,
    Solenoid,
    Superposition,
    Taylor,
)
from lattix.ir.fieldmap import replacement_for
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.rf import thin_gap_surrogate_length
from lattix.ir.walk import propagate

_THIN = 1e-12
_FRINGE_M = 1e-9              # pole-face fringe zone of a hard-edge dipole file
# The generated solenoid table straddles each nominal end with three intervals of width h (nodes at
# −1.5h, −0.5h, +0.5h, +1.5h): Bz/B = 0, _SOL_EDGE_A, 1 − _SOL_EDGE_A, 1 and Br/(B·r/h) = 0, −1/4,
# −1/4, 0.  Under IMPACT-T's bilinear (r, z) interpolation the paraxial transfer map of that table
# equals the hard-edge solenoid's to O(h²) (3e-8 at h = 0.2 mm for 0.4 m, 0.5 T, 2.1 MeV protons;
# a plain 0 → ½ → 1 ramp is O(h): 9e-6); ∫Bz is exact per end and the overshoot compensates the
# trapezoidal Br.  Found by minimising the model's map deviation (tools/impactt_solenoid_table.py).
_SOL_EDGE_A = -0.2318182
_THIN_SURROGATE_MIN = 1e-3    # the shortest surrogate cavity standing in for a thin gap


def _surrogate_length(V: float, phase_rad: float, freq: float, ref) -> float:
    return thin_gap_surrogate_length(V, phase_rad, freq, ref, _THIN_SURROGATE_MIN)
_FILE_COLUMN = {1: 2, 3: 2, 4: 3, 5: 3}       # 0-based index of the file id among a card's values (RF: 4)
#: value indices (0 = zedge) that hold a position for the run-control types, moved with the card
_POSITION_VALUES: dict[int, tuple[int, ...]] = {-1: (0, 1), -11: (0, 1), -2: (0, 2), -3: (0, 2), -4: (0, 2),
                                                -5: (0, 2), -9: (0, 2), -99: (0, 2)}


def _shift_card(c: _Card, dz: float) -> None:
    for i in _POSITION_VALUES.get(c.itype, (0,)):
        if i < len(c.values):
            c.values[i] += dz


def fmt(x: float) -> str:
    """``%.15g`` with ``-0`` normalised away (Fortran list-directed input reads it)."""
    if x == 0:
        x = 0.0
    return f"{x:.15g}"


@dataclass(frozen=True)
class Rule:
    target: str
    cls: str = "EXACT"
    code: str = "OK"
    message: str = ""


@dataclass
class _Card:
    length: float
    nseg: int
    mapstp: int
    itype: int
    values: list[float]
    name: str = ""
    kind: str = ""
    note: str = ""
    synthetic: bool = False           # a thin element given a length: absorbed by the neighbouring drifts
    rf: dict | None = None            # deferred amplitude/phase calibration (needs the final position)
    tag_extra: str = ""               # extra ``key=value`` pairs on the ``! lattix:`` tag line
    beta_in: float = 0.0              # reference velocities of a synthetic card (its time of flight)
    beta_out: float = 0.0

    @property
    def zedge(self) -> float:
        return self.values[0] if self.values else 0.0

    @zedge.setter
    def zedge(self, z: float) -> None:
        self.values[0] = z

    def render(self) -> str:
        vals = " ".join(fmt(v) for v in self.values)
        head = f"{fmt(self.length)} {self.nseg:d} {self.mapstp:d} {self.itype:d}"
        line = f"{head} {vals} /" if vals else f"{head} /"
        return f"{line} {self.note}" if self.note else line


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`; ``render`` gives the text."""

    format = "impactt"
    RULES: dict[str, Rule] = {
        "Drift": Rule("0 drift tube"),
        "Quadrupole": Rule("1 quadrupole (lab gradient T/m, hard edge)"),
        "Sextupole": Rule("5 multipole id 2 (B'' T/m²)"),
        "Octupole": Rule("5 multipole id 3 (B''' T/m³)"),
        "Multipole": Rule("-1 kick for the dipole terms, short type 1/5 elements for the rest", "EQUIVALENT",
                          "IMPACTT_THIN_MULTIPOLE_AS_THICK",
                          "IMPACT-T has no thin multipole: each order becomes a short element of the "
                          "integrated strength centred at the multipole"),
        "Bend": Rule("4 dipole (By, arc length, pole-face file)"),
        "Solenoid": Rule("3 solenoid (Bz0 T, radius 0 = hard edge)"),
        "RFCavity": Rule("104 SC cavity + rfdataN (raised-cosine Ez profile)", "EQUIVALENT", "IMPACTT_RF_PROFILE",
                         "the cavity is a type-104 element with a generated on-axis profile whose amplitude "
                         "and driven phase reproduce the reference gain q·V·cos φs"),
        "FieldMap": Rule("the lattix.ir.fieldmap replacement ladder"),
        "NCells": Rule("104 cavity with a cell-train profile", "EQUIVALENT", "NCELLS_AS_PROFILE",
                       "the cell train is written as a type-104 cavity whose profile alternates per cell"),
        "RFQCell": Rule("0 drift", "LOSSY", "RFQ_TO_DRIFT", "IMPACT-T has no RFQ cell; written as a drift"),
        "Kicker": Rule("-1 centroid momentum shift (dpx = hkick·γβ)"),
        "Collimator": Rule("-11 collimator"),
        "Marker": Rule("0 drift of zero length (tagged)"),
        "Instrument": Rule("0 drift of zero length (tagged)", "EQUIVALENT", "INSTRUMENT_AS_MARKER",
                           "IMPACT-T diagnostics are run controls; the monitor is a tagged zero-length drift"),
        "Foil": Rule("0 drift (tagged)", "LOSSY", "FOIL_TO_MARKER", "IMPACT-T has no stripping foil"),
        "Taylor": Rule("0 drift (tagged)", "DROPPED", "TAYLOR_DROPPED",
                       "IMPACT-T's -12 external map reads linearmap.in in its own units; not generated"),
        "Patch": Rule("0 drift (tagged)", "LOSSY", "PATCH_DROPPED", "IMPACT-T has no patch element"),
        "ReferenceChange": Rule("(nothing)", "LOSSY", "REFCHANGE_DROPPED",
                                "IMPACT-T's reference energy follows its own integration"),
        "Freq": Rule("(nothing)", "EXACT", "OK", "the RF clock lives on each cavity's frequency column"),
        "Directive": Rule("comment (or the native card)", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("children in order", "LOSSY", "SUPERPOSITION_FLATTENED",
                              "superposed maps written as consecutive elements"),
    }

    # ------------------------------------------------------------------ entry points
    def write(self, lattice: Lattice, path, *, strict: bool = False, write_rfdata: bool = True,
              **options) -> FidelityReport:
        path = Path(path)
        rep = FidelityReport(target_format="impactt", target_file=str(path))
        text, files = self.render(lattice, report=rep, **options)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if write_rfdata:
            for name, body in files.items():
                (path.parent / name).write_text(body)
        rep.raise_if(strict)
        return rep

    def render(self, lattice: Lattice, *, report: FidelityReport | None = None, dt_s: float | None = None,
               n_steps: int | None = None, n_particles: int | None = None, grid: tuple[int, int, int] | None = None,
               current_A: float | None = None, distribution: int | None = None, radius_m: float = 1.0,
               perdlen_m: float | None = None, flagdiag: int | None = None, thin_gap_length_m: float | None = None,
               thin_multipole_length_m: float = 1e-3, harmonics: int = 40, profile_points: int = 201,
               solenoid_ramp_m: float = 2e-4, first_rfdata_id: int = 1, errors: bool | None = None,
               use: str | None = None) -> tuple[str, dict[str, str]]:
        """The deck text and the ``rfdataN`` files it references (``{name: content}``).

        The run settings come from ``meta['impactt_header']`` when the lattice was read from
        IMPACT-T, else from the keyword defaults (``dt_s`` 1 ps, steps for 1.2× the reference
        time of flight, 1000 particles, 32³ mesh, a Gaussian distribution, 0 A).
        """
        rep = report if report is not None else FidelityReport(target_format="impactt")
        self.rep = rep
        self.lat = lattice
        self.opts = dict(radius_m=radius_m, thin_gap_length_m=thin_gap_length_m,
                         thin_multipole_length_m=thin_multipole_length_m, harmonics=harmonics,
                         profile_points=profile_points, solenoid_ramp_m=solenoid_ramp_m)
        self.files: dict[str, str] = {}
        self._by_hash: dict[str, int] = {}
        self._next_id = int(first_rfdata_id)
        lattice, _ = with_rf_focusing(lattice, "impactt")
        placed = propagate(lattice, use)
        self.s_offset = float(lattice.meta.get("impactt_s_offset") or 0.0)
        # a source deck with overlapping cards has no sequential picture: every card that came from
        # it keeps its own zedge (the IR's s only orders them)
        self.native_positions = any((p.element.native.get("impactt") or {}).get("pushed") for p in placed)
        self.errors = (any(_has_shift(p.element) for p in placed) if errors is None else bool(errors))
        cards: list[_Card] = []
        comments: list[tuple[int, str]] = []
        for p in placed:
            for c in self._element(p, comments, len(cards)):
                cards.append(c)
        self._absorb_synthetic(cards)
        self._shift_to_positive(cards, lattice)
        self._calibrate_rf(cards)
        total = max([c.zedge + c.length for c in cards] + [self.s_offset])
        cards.append(_Card(0.0, 1, 1, -99, [total, 1.0, total], note="! end of the beamline"))
        header = self._header(lattice, placed, dt_s=dt_s, n_steps=n_steps, n_particles=n_particles, grid=grid,
                              current_A=current_A, distribution=distribution, radius_m=radius_m,
                              perdlen_m=perdlen_m, flagdiag=flagdiag)
        return self._render(lattice, header, cards, comments, placed), dict(self.files)

    # ------------------------------------------------------------------ header
    def _header(self, lat: Lattice, placed: list[Placed], **kw) -> Header:
        meta = lat.meta.get("impactt_header")
        h = Header.from_meta(meta) if meta else Header()
        ref = lat.reference
        h.kinetic_energy_eV = float(ref.kinetic_energy_eV)
        h.mass_eV = float(ref.species.mass_eV)
        h.charge = float(ref.species.charge)
        h.frequency_Hz = self._frequency(lat, placed)
        h.flagerr = 1 if self.errors else 0
        if kw.get("dt_s") is not None:
            h.dt_s = float(kw["dt_s"])
        elif not meta:
            h.dt_s = 1e-12
        if kw.get("n_particles") is not None:
            h.np = int(kw["n_particles"])
        if kw.get("grid") is not None:
            h.nx, h.ny, h.nz = (int(v) for v in kw["grid"])
        if kw.get("current_A") is not None:
            h.current_A = float(kw["current_A"])
        if kw.get("distribution") is not None:
            h.flagdist = int(kw["distribution"])
        if kw.get("flagdiag") is not None:
            h.flagdiag = int(kw["flagdiag"])
        if kw.get("perdlen_m") is not None:
            h.perdlen = float(kw["perdlen_m"])
        if not meta:
            h.xrad = h.yrad = float(kw.get("radius_m") or 1.0)
        if kw.get("n_steps") is not None:
            h.ntstep = int(kw["n_steps"])
        elif not meta:
            tof = placed[-1].ref_out.time_s if placed and placed[-1].ref_out is not None else 0.0
            h.ntstep = int(math.ceil(1.2 * tof / h.dt_s)) + 1000
        return h

    def _frequency(self, lat: Lattice, placed: list[Placed]) -> float:
        if lat.reference.rf_frequency_Hz:
            return float(lat.reference.rf_frequency_Hz)
        for p in placed:
            if isinstance(p.element, Freq) and p.element.frequency_Hz:
                return float(p.element.frequency_Hz)
            rf = getattr(p.element, "rf", None)
            if rf is not None and rf.frequency_Hz:
                return float(rf.frequency_Hz)
        meta = lat.meta.get("impactt_header") or {}
        if meta.get("frequency_Hz"):
            return float(meta["frequency_Hz"])
        self.rep.lossy("IMPACTT_NO_FREQUENCY",
                       "no RF frequency anywhere in the lattice; the header's scale frequency defaults to 1 GHz")
        return 1e9

    def _render(self, lat: Lattice, h: Header, cards: list[_Card], comments: list[tuple[int, str]],
                placed: list[Placed]) -> str:
        d = h.distparam if len(h.distparam) == 21 else [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] * 3
        rows = [" ".join(fmt(v) for v in d[7 * i:7 * i + 7]) for i in range(3)]
        out = [
            f"! IMPACT-T deck written by lattix {__version__}",
            f"! lattice: {lat.name!r}",
            "! header: 9 records (comment lines starting with ! are skipped)",
            "! 1: npcol nprow",
            f"{h.npcol} {h.nprow}",
            "! 2: dt[s] ntstep nbunch",
            f"{fmt(h.dt_s)} {h.ntstep} {h.nbunch}",
            "! 3: dim np flagmap flagerr flagdiag flagimg zimage[m]",
            f"{h.dim} {h.np} {h.flagmap} {h.flagerr} {h.flagdiag} {h.flagimg} {fmt(h.zimage)}",
            "! 4: nx ny nz flagbc xrad[m] yrad[m] zleng[m]",
            f"{h.nx} {h.ny} {h.nz} {h.flagbc} {fmt(h.xrad)} {fmt(h.yrad)} {fmt(h.perdlen)}",
            "! 5: flagdist rstartflg flagsbstp nemission temission[s]",
            f"{h.flagdist} {h.rstartflg} {h.flagsbstp} {h.nemission} {fmt(h.temission)}",
            "! 6-8: distparam per plane (sig sigp mu scale pscale mu1 mu2)",
            rows[0], rows[1], rows[2],
            "! 9: current[A] kinetic_energy[eV] mass[eV] charge[e] frequency[Hz] phase[rad]",
            f"{fmt(h.current_A)} {fmt(h.kinetic_energy_eV)} {fmt(h.mass_eV)} {fmt(h.charge)} "
            f"{fmt(h.frequency_Hz)} {fmt(h.phase_ini_rad)}",
            "!",
            "! lattice: length nseg mapstp type zedge value2 … / (a card ends at the '/')",
        ]
        by_index: dict[int, list[str]] = {}
        for idx, txt in comments:
            by_index.setdefault(idx, []).append(txt)
        for i, c in enumerate(cards):
            for txt in by_index.get(i, ()):
                out.append(txt)
            if c.name:
                extra = f" {c.tag_extra}" if c.tag_extra else ""
                out.append(f"! lattix: name={quote(c.name, safe='')} kind={c.kind}{extra}")
            out.append(c.render())
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------ helpers
    def _record(self, el: Element, rule: Rule, **details) -> None:
        if rule.cls == "EXACT":
            self.rep.exact(el.name, el.kind, code=rule.code, message=rule.message)
        else:
            self.rep.add(rule.cls, rule.code, rule.message, element=el.name, kind=el.kind, **details)

    def _radius(self, el: Element) -> float:
        ap = el.aperture
        if ap is not None:
            r = [abs(v) for lim in (ap.x_limits, ap.y_limits) if lim for v in lim]
            if r and min(r) > 0:
                return min(r)
        return float(self.opts["radius_m"])

    def _shift_cols(self, el: Element) -> list[float]:
        s = el.shift
        if s is None or s.is_zero():
            return [0.0] * 5
        if not self.errors:
            self.rep.lossy("IMPACTT_MISALIGNMENT_DROPPED",
                           "the deck's flagerr is 0, so IMPACT-T would ignore the misalignment columns; written "
                           "as zeros (pass errors=True to apply them)", element=el.name, kind=el.kind)
            return [0.0] * 5
        if s.z_offset:
            self.rep.lossy("IMPACTT_NO_Z_OFFSET", "IMPACT-T misalignments have no longitudinal offset",
                           element=el.name, kind=el.kind, z_offset=s.z_offset)
        return [s.x_offset, s.y_offset, s.x_rot, s.y_rot, s.tilt]

    def _nseg(self, el: Element) -> tuple[int, int]:
        return int(el.tracking.get("nseg", 1) or 1), int(el.tracking.get("mapstp", 1) or 1)

    def _file(self, values: list[float], *, prefer: int | None = None) -> int:
        """Register an ``rfdataN`` body (one number per line), re-using an identical one."""
        return self._file_text("\n".join(fmt(v) for v in values) + "\n", prefer=prefer)

    def _file_rows(self, rows: list[list[float]], *, table: bool = False, prefer: int | None = None) -> int:
        return self._file_text("\n".join(" ".join(fmt(v) for v in r) for r in rows) + "\n", table=table,
                               prefer=prefer)

    def _file_text(self, body: str, *, table: bool = False, prefer: int | None = None) -> int:
        """Register a data file, re-using an identical one: ``rfdataN`` (Fourier coefficients, pole
        faces) or, for a solenoid's (r, z) table, ``1TN.T7`` (``read2tsol_Data``'s name).  A file
        that came from an IMPACT-T deck keeps its id (``prefer``) when that id is still free."""
        h = hashlib.sha256((("T7" if table else "rf") + body).encode()).hexdigest()
        if h in self._by_hash:
            return self._by_hash[h]
        used = set(self._by_hash.values())
        if prefer and prefer > 0 and prefer not in used:
            fid = int(prefer)
        else:
            while self._next_id in used:
                self._next_id += 1
            fid = self._next_id
            self._next_id += 1
        if fid > 999:
            self.rep.lossy("IMPACTT_RFDATA_LIMIT", "IMPACT-T reads at most 999 data files of a kind (Data.f90)")
        self.files[solenoid_table_name(fid) if table else rfdata_name(fid)] = body
        self._by_hash[h] = fid
        return fid

    def _z(self, p: Placed) -> float:
        """Absolute starting edge of a placed element (the IR's s plus the source deck's origin)."""
        return p.s_in + self.s_offset

    def _drift_card(self, el: Element, zedge: float, length: float | None = None, kind: str | None = None) -> _Card:
        L = el.length if length is None else length
        nseg, mapstp = self._nseg(el)
        return _Card(L, nseg, mapstp, 0, [zedge, self._radius(el)], kind=kind or "Drift")

    # ------------------------------------------------------------------ dispatch
    def _element(self, p: Placed, comments: list[tuple[int, str]], card_index: int) -> list[_Card]:
        el = p.element
        self._s_in = p.s_in
        rule = self.RULES.get(el.kind)
        if rule is None:                                      # pragma: no cover - RULES is total
            raise KeyError(f"the IMPACT-T writer has no rule for kind {el.kind!r}")
        native = el.native.get("impactt") or {}
        if native.get("implicit"):
            self.rep.exact(el.name, el.kind, code="IMPACTT_IMPLICIT_GAP",
                           message="a gap between IMPACT-T cards; nothing is written for it")
            return []
        if isinstance(el, Freq):
            self._record(el, rule)
            return []
        if isinstance(el, Directive):
            if native.get("passthrough"):
                return self._passthrough(el, native)
            self._record(el, rule, role=el.role, card=el.card)
            return []
        if native.get("passthrough"):
            return self._passthrough(el, native)
        fn = getattr(self, f"_w_{el.kind.lower()}")
        cards = fn(el, p, rule)
        for c in cards:
            c.name = c.name or el.name
            c.kind = c.kind or el.kind
        if self.native_positions and native.get("values") and cards:
            # the source deck had overlapping cards: keep this card's own position rather than the IR's
            dz = float(native["values"][0]) - cards[0].zedge
            for c in cards:
                _shift_card(c, dz)
            self.rep.equivalent("IMPACTT_NATIVE_POSITION",
                                "the card keeps the zedge of the source deck, whose cards overlap",
                                element=el.name, kind=el.kind)
        return cards

    def _passthrough(self, el: Element, native: dict) -> list[_Card]:
        self.rep.exact(el.name, el.kind, code="IMPACTT_NATIVE_PASSTHROUGH",
                       message=f"type {native['type']} re-emitted from native['impactt']")
        c = _Card(el.length, int(native["nseg"]), int(native["mapstp"]), int(native["type"]),
                  list(native["values"]), name=el.name, kind=el.kind)
        if not self.native_positions and c.values:
            _shift_card(c, (self._s_in + self.s_offset) - c.values[0])
        coefs = el.meta.get("impactt_rfdata")
        col = _FILE_COLUMN.get(c.itype, 4 if c.itype >= 100 else None)
        if coefs is not None and col is not None and len(c.values) > col and c.values[col] > 0:
            # its data file travels with it, under its own id when that is free
            c.values[col] = float(self._file(list(coefs), prefer=int(el.meta.get("impactt_file_id") or 0)))
        return [c]

    def _w_drift(self, el, p, rule) -> list[_Card]:
        if el.length < 0.0:
            self.rep.equivalent("NEGATIVE_DRIFT_DROPPED", "IMPACT-T places every card by its absolute zedge: a "
                                "negative drift only moves the following cards back and is not written",
                                element=el.name, kind="Drift", length_m=float(el.length))
            return []
        self._record(el, rule)
        return [self._drift_card(el, self._z(p))]

    def _w_quadrupole(self, el, p, rule) -> list[_Card]:
        mp = el.multipole
        g, tilt = float(mp.Bn.get(1, 0.0)), float(mp.tilt.get(1, 0.0))
        bs = float(mp.Bs.get(1, 0.0))
        if bs:
            g, tilt = math.hypot(g, bs), tilt + 0.5 * math.atan2(bs, g)
            self.rep.exact(el.name, "Quadrupole", code="IMPACTT_SKEW_AS_ROTATION",
                           message="the skew component is written as a roll of the normal gradient")
        if any(v for k, v in mp.Bn.items() if k != 1) or any(v for k, v in mp.Bs.items() if k != 1):
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED",
                           "IMPACT-T's quadrupole carries the gradient only; other orders dropped",
                           element=el.name, kind="Quadrupole")
        self._record(el, rule)
        sh = self._shift_cols(el)
        if tilt:
            if not self.errors:
                self.rep.lossy("IMPACTT_TILT_NEEDS_ERRORS",
                               "the quadrupole roll lives in the rotation-error column, which IMPACT-T applies "
                               "only with flagerr = 1 (pass errors=True)", element=el.name, kind="Quadrupole")
            sh[4] += tilt
        nseg, mapstp = self._nseg(el)
        return [_Card(el.length, nseg, mapstp, 1, [self._z(p), g, 0.0, self._radius(el), *sh, 0.0, 0.0])]

    def _multipole_card(self, el, zedge: float, length: float, order: int, strength: float, rot: float,
                        ) -> _Card:
        nseg, mapstp = self._nseg(el)
        sh = self._shift_cols(el)
        sh[4] += rot
        return _Card(length, nseg, mapstp, 5, [zedge, float(order), strength, 0.0, self._radius(el), *sh])

    def _w_sextupole(self, el, p, rule) -> list[_Card]:
        return self._thick_multipole(el, p, rule, 2)

    def _w_octupole(self, el, p, rule) -> list[_Card]:
        return self._thick_multipole(el, p, rule, 3)

    def _thick_multipole(self, el, p, rule, order: int) -> list[_Card]:
        mp = el.multipole
        bn, bs = float(mp.Bn.get(order, 0.0)), float(mp.Bs.get(order, 0.0))
        rot = float(mp.tilt.get(order, 0.0))
        if bs:
            bn, rot = math.hypot(bn, bs), rot + math.atan2(bs, bn) / order
        if any(v for k, v in mp.Bn.items() if k != order) or any(v for k, v in mp.Bs.items() if k != order):
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "IMPACT-T's multipole carries one order; the others dropped",
                           element=el.name, kind=el.kind)
        self._record(el, rule)
        return [self._multipole_card(el, self._z(p), el.length, order, bn, rot)]

    def _w_multipole(self, el: Multipole, p: Placed, rule) -> list[_Card]:
        ref = p.ref_in or self.lat.reference
        mp = el.multipole
        brho = ref.brho_signed
        bg = ref.beta * ref.gamma
        cards: list[_Card] = []
        centre = self._z(p) + 0.5 * el.length
        bn0, bs0 = float(mp.BnL.get(0, 0.0)), float(mp.BsL.get(0, 0.0))
        if bn0 or bs0:
            hk, vk = -bn0 / brho, bs0 / brho
            cards.append(_Card(0.0, 1, 1, -1, [centre, centre, 0.0, hk * bg, 0.0, vk * bg, 0.0, 0.0],
                               kind="Kicker"))
            self.rep.lossy("IMPACTT_MULTIPOLE_AS_KICK", "a thin multipole's dipole terms become a -1 kick; the "
                                                        "multipole identity does not survive a read-back",
                           element=el.name, kind="Multipole")
        ell = float(self.opts["thin_multipole_length_m"])
        for order in sorted(set(mp.BnL) | set(mp.BsL)):
            if order == 0:
                continue
            bn, bs = float(mp.BnL.get(order, 0.0)), float(mp.BsL.get(order, 0.0))
            if not (bn or bs):
                continue
            rot = float(mp.tilt.get(order, 0.0)) + (math.atan2(bs, bn) / order if bs else 0.0)
            strength = math.hypot(bn, bs) / ell
            if order == 1:
                sh = self._shift_cols(el)
                sh[4] += rot
                c = _Card(ell, 1, 1, 1, [centre - 0.5 * ell, strength, 0.0, self._radius(el), *sh, 0.0, 0.0],
                          kind="Multipole", synthetic=True, beta_in=(p.ref_in or self.lat.reference).beta,
                          beta_out=(p.ref_out or p.ref_in or self.lat.reference).beta)
            elif order in (2, 3, 4):
                c = self._multipole_card(el, centre - 0.5 * ell, ell, order, strength, rot)
                c.kind, c.synthetic = "Multipole", True
            else:
                self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", f"IMPACT-T has no multipole of order {order}",
                               element=el.name, kind="Multipole")
                continue
            cards.append(c)
        self._record(el, rule, length_m=ell)
        if el.length > _THIN:
            cards.insert(0, self._drift_card(el, self._z(p), kind="Multipole"))
        if not cards:
            cards.append(self._drift_card(el, self._z(p), 0.0, kind="Multipole"))
        return cards

    def _w_bend(self, el: Bend, p: Placed, rule) -> list[_Card]:
        ref = p.ref_in or self.lat.reference
        b = el.bend
        brho = ref.brho_signed
        if abs(b.tilt_ref) > 1e-12:
            self.rep.lossy("IMPACTT_BEND_TILT_DROPPED",
                           "the tracked dipole field uses By only: the bend is written in the horizontal plane",
                           element=el.name, kind="Bend", tilt_ref=b.tilt_ref)
        if el.multipole.Bn.get(1) or el.multipole.Bs.get(1):
            self.rep.lossy("MULTIPOLE_ORDERS_DROPPED", "IMPACT-T's dipole has no gradient",
                           element=el.name, kind="Bend")
        if (b.edge_int1 or b.edge_int2) and b.hgap:
            self.rep.lossy("BEND_FRINGE_DROPPED", "the pole-face file is written with hard edges (fint·hgap lost)",
                           element=el.name, kind="Bend")
        if el.length <= _THIN or b.angle == 0.0:
            self.rep.equivalent("ZERO_ANGLE_BEND_AS_DRIFT", "zero-angle bend written as a drift",
                                element=el.name, kind="Bend")
            return [self._drift_card(el, self._z(p), kind="Bend")]
        angle, arc = float(b.angle), float(el.length)
        rho = arc / abs(angle)
        by = brho * angle / arc
        s = 1.0 if angle > 0 else -1.0
        xe, ze = -s * rho * (1.0 - math.cos(angle)), rho * math.sin(abs(angle))
        # the pole-face lines of a negative bend are the mirror (x → −x) of the positive one with the faces
        # negated (MAD-X: (angle<0, e1, e2) ≡ (angle>0, tilt π, −e1, −e2)): slope k1 = tan(e1) either way,
        # exit slope k4 = s·tan(|θ| − s·e2)
        k1 = math.tan(b.e1)
        k4 = s * math.tan(abs(angle) - s * b.e2)
        b4 = ze - k4 * xe
        native = el.meta.get("impactt_bend")
        if native and abs(native.get("angle", 0.0) - angle) < 1e-15 and native.get("e1") == b.e1 \
                and native.get("e2") == b.e2 and el.meta.get("impactt_rfdata"):
            fid = self._file(list(el.meta["impactt_rfdata"]))
        else:
            fid = self._file([0.0, ref.gamma, k1, 0.0, k1, _FRINGE_M, k4, b4 - _FRINGE_M, k4, b4,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, arc])
        self._record(el, rule)
        nseg, mapstp = self._nseg(el)
        return [_Card(arc, nseg, mapstp, 4, [self._z(p), 0.0, by, float(fid), self._radius(el), *self._shift_cols(el)])]

    def _w_solenoid(self, el: Solenoid, p, rule) -> list[_Card]:
        """Type 3 with a generated ``(r, z)`` table (``getfldt_Sol`` reads no analytic field): a
        normalised flat top whose ends are the three-interval straddling edges of ``_SOL_EDGE_A``
        (Bz overshoot plus the Br hat carrying the whole ``−(r/2)·B`` edge kick), so that the
        tracked map is the hard-edge solenoid's to second order in the table interval."""
        L, B = float(el.length), float(el.solenoid.Bsol_T)
        native = el.meta.get("impactt_table")
        nseg, mapstp = self._nseg(el)
        rmax = self._radius(el)                    # the table's radial extent (particles beyond it stop)
        if L <= _THIN:
            self._record(el, rule)
            return [self._drift_card(el, self._z(p), 0.0, kind="Solenoid")]
        if native and native.get("length") and el.meta.get("impactt_rfdata_rows") is not None:
            fid = self._file_rows(el.meta["impactt_rfdata_rows"], table=True, prefer=int(native.get("file_id") or 0))
            pad, lt, b0 = float(native["pad"]), float(native["length"]), float(native["Bz0"])
        else:
            h = float(self.opts["solenoid_ramp_m"])
            n_flat = max(1, int(round((L - 3.0 * h) / h)))
            h = L / (n_flat + 3)                  # the plateau ends 1.5 h inside the nominal ends
            nz = n_flat + 6
            lt = L + 3.0 * h
            pad = round(1.5 * h, 12)              # on the picometre grid: the tag adds up to the nominal end
            b0 = B
            edge = (0.0, _SOL_EDGE_A, 1.0 - _SOL_EDGE_A)
            hat = 0.25 * rmax / h                 # Br at r = rmax on the two inner edge nodes
            rows = [f"0 {fmt(rmax * 100.0)} 1", f"0 {fmt(lt * 100.0)} {nz}"]
            for j in range(nz + 1):
                if j < 3:
                    bz, brj = edge[j], (-hat if j else 0.0)
                elif j > nz - 3:
                    bz, brj = edge[nz - j], (hat if j < nz else 0.0)
                else:
                    bz, brj = 1.0, 0.0
                rows.append("0 " + fmt(bz))                       # r = 0
                rows.append(fmt(brj) + " " + fmt(bz))             # r = rmax
            fid = self._file_text("\n".join(rows) + "\n", table=True)
            self.rep.equivalent("IMPACTT_SOLENOID_TABLE",
                                f"the hard-edge solenoid is a type-3 element with a generated (r, z) table "
                                f"({lt:.6g} m, straddling edges of 3 × {h:.3g} m) whose tracked map equals the "
                                f"hard-edge map to second order in the table interval",
                                element=el.name, kind="Solenoid", interval_m=h, table_length_m=lt)
        self._record(el, rule)
        c = _Card(lt, nseg, mapstp, 3, [self._z(p) - pad, b0, float(fid), rmax, *self._shift_cols(el)])
        c.tag_extra = f"L={fmt(L)} Bsol={fmt(B)} pad={fmt(pad)}"
        return [c]

    def _profile_card(self, el: Element, p: Placed, rule, *, V: float, freq: float, L_elem: float,
                      shape: tuple, active: float, phase: float, sync: bool, thin: bool, adaptive: bool,
                      zedge: float, extra_details: dict | None = None) -> list[_Card]:
        """A type-104 card whose ``rfdataN`` profile (``shape``: ``("bump", window, None)`` or
        ``("cells", n_cell, pi_mode)``) and driven phase are produced once its final length and
        position are known (``_absorb_synthetic``, ``_calibrate_rf``)."""
        ref = p.ref_in or self.lat.reference
        if not sync:
            self.rep.equivalent("IMPACTT_RF_PHASE_NOT_SYNC",
                                "the source gave a driven RF phase; written so that the reference gain still "
                                "equals V·cos(phase)", element=el.name, kind=el.kind)
        rfm = el.meta.get("impactt_rf")
        rf = getattr(el, "rf", None)
        fixed = None
        coefs = None
        fid = 0
        if rfm and rf is not None and rfm.get("voltage_V") == rf.voltage_V and rfm.get("phase_rad") == rf.phase_rad \
                and el.meta.get("impactt_rfdata"):
            coefs = list(el.meta["impactt_rfdata"])
            fixed = (float(rfm["scale"]), float(rfm["theta0_deg"]))
            fid = self._file(coefs, prefer=int(el.meta.get("impactt_file_id") or 0))
        self._record(el, rule, voltage_V=V, length_m=L_elem, **(extra_details or {}))
        self.rep.equivalent("IMPACTT_RF_DRIVEN_PHASE",
                            "theta0 is a driven RF phase computed from the lattix reference time of flight; "
                            "IMPACT-T re-integrates its own", element=el.name, kind=el.kind)
        nseg, mapstp = self._nseg(el)
        # the walk's time applies at s_in (the gap position for a thin cavity); the card's entrance
        # is only known once the synthetic length has been placed among the neighbouring drifts
        calib = {"coefs": coefs, "shape": shape, "active": active, "V": V, "phase": phase, "freq": freq,
                 "beta": ref.beta, "ke": ref.kinetic_energy_eV, "mass": ref.species.mass_eV,
                 "charge": float(ref.species.charge), "t0": ref.time_s, "s0": self._z(p), "fixed": fixed,
                 "wanted": L_elem, "adaptive": adaptive, "centre": self._z(p)}
        ref_out = p.ref_out or ref
        return [_Card(L_elem, nseg, mapstp, 104, [zedge, 0.0, freq, 0.0, float(fid), self._radius(el),
                                                  *self._shift_cols(el)], synthetic=thin, rf=calib,
                      beta_in=ref.beta, beta_out=ref_out.beta)]

    def _shift_to_positive(self, cards: list[_Card], lattice: Lattice) -> None:
        """IMPACT-T applies no field at z < 0 (MEASURED: a solenoid table starting at −0.3 mm lost 45 %
        of its entrance kick): a deck whose first field region would start below zero is moved so that
        it starts at zero; the beam then flies that far field-free before the nominal start, which the
        driven phases account for (the deck's time origin is the reference at z = 0)."""
        self.shift_time = 0.0
        start = min((c.zedge for c in cards if c.values), default=0.0)
        if start >= 0.0 or lattice.meta.get("impactt_s_offset") is not None:
            return                                # a deck that came from IMPACT-T keeps its own origin
        shift = -start
        for c in cards:
            _shift_card(c, shift)
            if c.rf is not None:
                c.rf["s0"] += shift
        ref = lattice.reference
        self.shift_time = shift / (ref.beta * C_LIGHT)
        self.s_offset += shift
        self.rep.equivalent("IMPACTT_DECK_OFFSET",
                            f"the deck starts {shift:.6g} m after z = 0 so that its first field region (a table "
                            "that ramps before its nominal end) lies at z ≥ 0, where IMPACT-T evaluates fields; "
                            "the reference is taken to be at z = 0 at t = 0", shift_m=shift)

    def _calibrate_rf(self, cards: list[_Card]) -> None:
        """Driven phases in card order.  IMPACT-T's reference gains its energy along each profile,
        so it leaves every cavity a little earlier or later than the IR's walk (a thin gap at a point,
        a thick cavity crossed at its entrance velocity) says: the difference is carried to the
        cavities downstream (MEASURED: 2.4e-4 on the MEBT's energy without it, 1e-7 with)."""
        t_corr = 0.0                                  # IMPACT-T's clock minus the IR walk's, so far
        for c in cards:
            if c.rf is None:
                continue
            k = c.rf
            if k["fixed"] is not None:
                c.values[1], c.values[3] = k["fixed"]
                continue
            if k.get("coefs") is None:
                # the profile of a surrogate depends on the length the neighbouring drifts gave it
                kind, a, b = k["shape"]
                if kind == "cells":
                    z, ez = cell_train(c.length, a, pi_mode=b, active=min(k["active"], c.length))
                else:
                    z, ez = raised_cosine(c.length, min(a, c.length), int(self.opts["profile_points"]))
                peak = max(abs(v) for v in ez) or 1.0
                k["coefs"] = fourier_coefficients(z, [v / peak for v in ez], c.length, int(self.opts["harmonics"]))
                c.values[4] = float(self._file(k["coefs"]))
            t_in = k["t0"] + self.shift_time + t_corr + (c.zedge - k["s0"]) / (k["beta"] * C_LIGHT)
            c.values[1], c.values[3] = calibrate(k["coefs"], c.length, k["V"], k["phase"], k["freq"], t_in,
                                                 k["ke"], k["mass"], k["charge"])
            _, t_out = integrate_reference(k["coefs"], c.length, c.values[1], c.values[3], k["freq"], t_in,
                                           k["ke"], k["mass"], k["charge"])
            beta_out = c.beta_out or k["beta"]
            if c.synthetic:                          # the IR: a point gain at s0 (time t0), then β_out
                t_ir_exit = k["t0"] + self.shift_time + t_corr + (c.zedge + c.length - k["s0"]) / (beta_out * C_LIGHT)
            else:                                    # the IR crosses a thick cavity at its entrance velocity
                t_ir_exit = t_in + c.length / (k["beta"] * C_LIGHT)
            t_corr += t_out - t_ir_exit

    def _w_rfcavity(self, el: RFCavity, p: Placed, rule) -> list[_Card]:
        ref = p.ref_in or self.lat.reference
        rf = el.rf
        V = rf.voltage_V
        if not V and rf.gradient_V_per_m is not None:
            V = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None else el.length)
        if rf.dE_ref_eV is not None and rf.dE_ref_eV and abs(math.cos(rf.phase_rad)) > 1e-12 and not V:
            V = rf.dE_ref_eV / math.cos(rf.phase_rad)
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        if not freq or not V:
            self.rep.lossy("IMPACTT_CAVITY_TO_DRIFT",
                           "an RF cavity without a frequency or a voltage cannot be written; replaced by a drift",
                           element=el.name, kind="RFCavity", voltage_V=V, frequency_Hz=freq)
            return [self._drift_card(el, self._z(p), kind="RFCavity")]
        thin = el.length <= _THIN
        beta_lambda = ref.beta * C_LIGHT / freq
        meta_thin = (el.meta.get("impactt_thin") or {}) if thin else {}
        # a thin gap becomes a short bump whose length balances the two ways a field integration
        # departs from the thin-gap model (MEASURED against HELIX): the kick is smeared over the
        # bump (∝ length: the MEBT's 80 kV gaps 6e-3 at 1–4 mm, 1.6e-2 at 12 mm, 3.6e-2 at βλ/2) and
        # the ponderomotive focusing of a strong short bump (∝ V²/length: a 577 kV DTL gap has 5×
        # the thin-gap focusing at 1 mm, 4 % at 22 mm); the drifts around it give the length up, so
        # it shrinks to what they hold (never below 1 mm)
        adaptive = thin and not meta_thin.get("length") and not self.opts["thin_gap_length_m"]
        L_elem = float(meta_thin.get("length") or self.opts["thin_gap_length_m"]
                       or _surrogate_length(V, rf.phase_rad, freq, ref)) if thin else el.length
        L_active = min(max(rf.L_active_m or 0.0, 0.0) or L_elem, L_elem)
        n_cell = int(rf.n_cell or 0)
        if n_cell > 1 and not thin:
            # a multi-cell cavity: half-wave cells over the active length (π mode unless travelling wave)
            shape: tuple = ("cells", n_cell, rf.cavity_type != "TRAVELING_WAVE")
            window = L_active
        else:
            # one accelerating bump: no wider than βλ/2, or the transit-time factor collapses (a 0.2 m
            # bump at 325 MHz for a 2.1 MeV proton spans 3 βλ and the calibrated field reflected it)
            window = min(L_active, 0.5 * beta_lambda)
            shape = ("bump", window, None)
        zedge = self._z(p) - 0.5 * L_elem if thin else self._z(p)
        return self._profile_card(el, p, rule, V=V, freq=freq, L_elem=L_elem, shape=shape, active=L_active,
                                  phase=rf.phase_rad, sync=rf.phase_is_sync, thin=thin, adaptive=adaptive,
                                  zedge=zedge, extra_details={"active_m": window, "n_cell": n_cell or 1})

    def _w_fieldmap(self, el: FieldMap, p: Placed, rule) -> list[_Card]:
        r = replacement_for(el)
        cards: list[_Card] = []
        s = self._z(p)
        for part in r.parts:
            sub = p.model_copy(update={"element": part, "s_in": s, "s_out": s + part.length})
            fn = getattr(self, f"_w_{part.kind.lower()}")
            for c in fn(part, sub, self.RULES[part.kind]):
                c.name, c.kind = c.name or el.name, c.kind or "FieldMap"
                cards.append(c)
            s += part.length
        getattr(self.rep, r.cls.lower())(r.code, r.message, element=el.name, kind="FieldMap", **r.details)
        for cls, code, msg in r.extra:
            getattr(self.rep, cls.lower())(code, msg, element=el.name, kind="FieldMap")
        return cards

    def _w_ncells(self, el: NCells, p: Placed, rule) -> list[_Card]:
        ref = p.ref_in or self.lat.reference
        rf = el.rf
        V = rf.voltage_V or (rf.gradient_V_per_m or 0.0) * el.length
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        n_cell = int(el.params.get("n_cells") or rf.n_cell or 1)
        if not (V and freq and el.length > 0):
            self.rep.lossy("NCELLS_TO_DRIFT", "an NCells without a voltage, a frequency or a length is written "
                                              "as a drift", element=el.name, kind="NCells")
            return [self._drift_card(el, self._z(p), kind="NCells")]
        mode = el.params.get("mode")
        shape = ("cells", n_cell, mode != 0)
        unmapped = {k: v for k, v in el.params.items()
                    if k in ("beta_g", "k_eot_i", "k_eot_o", "dz_i", "dz_o", "p_flag") and v}
        if unmapped:
            self.rep.lossy("IMPACTT_NCELLS_PARAMS",
                           "the cell-train profile has no geometric β_g, entrance/exit half-cell corrections or "
                           "TTF tail; those NCELLS columns are dropped", element=el.name, kind="NCells", **unmapped)
        return self._profile_card(el, p, rule, V=V, freq=freq, L_elem=el.length, shape=shape, active=el.length,
                                  phase=rf.phase_rad,
                                  sync=rf.phase_is_sync, thin=False, adaptive=False, zedge=self._z(p),
                                  extra_details={"n_cells": n_cell, "mode": mode})

    def _w_rfqcell(self, el: RFQCell, p, rule) -> list[_Card]:
        self._record(el, rule, length=el.length)
        return [self._drift_card(el, self._z(p), kind="RFQCell")]

    def _w_kicker(self, el: Kicker, p: Placed, rule) -> list[_Card]:
        ref = p.ref_in or self.lat.reference
        bg = ref.beta * ref.gamma
        if el.electric:
            self.rep.lossy("EKICK_AS_MAGNETIC", "IMPACT-T's -1 shift is species-independent; the electric flag is "
                                                "dropped", element=el.name, kind="Kicker")
        self._record(el, rule)
        centre = self._z(p) + 0.5 * el.length
        cards = [_Card(0.0, 1, 1, -1, [centre, centre, 0.0, el.hkick * bg, 0.0, el.vkick * bg, 0.0, 0.0])]
        if el.length > _THIN:
            self.rep.equivalent("THICK_KICKER_SPLIT", "the kick sits at the centre of a drift of the kicker's length",
                                element=el.name, kind="Kicker")
            cards.insert(0, self._drift_card(el, self._z(p), kind="Kicker"))
        return cards

    def _w_collimator(self, el: Collimator, p: Placed, rule) -> list[_Card]:
        ap = el.aperture
        r = self._radius(el)
        xl = ap.x_limits if ap and ap.x_limits else (-r, r)
        yl = ap.y_limits if ap and ap.y_limits else (-r, r)
        flag = 11.0 if (ap is not None and ap.shape == "ELLIPTICAL") else 1.0
        self._record(el, rule)
        centre = self._z(p) + 0.5 * el.length
        cards = [_Card(0.0, 1, 1, -11, [centre, centre, xl[0], xl[1], yl[0], yl[1], flag])]
        if el.length > _THIN:
            self.rep.equivalent("THICK_COLLIMATOR_AT_CENTRE", "IMPACT-T's -11 collimator is a plane: it sits at the "
                                                              "centre of a drift of the collimator's length",
                                element=el.name, kind="Collimator")
            cards.insert(0, self._drift_card(el, self._z(p), kind="Collimator"))
        return cards

    def _w_marker(self, el: Marker, p, rule) -> list[_Card]:
        self._record(el, rule)
        return [self._drift_card(el, self._z(p), 0.0, kind="Marker")]

    def _w_instrument(self, el: Instrument, p, rule) -> list[_Card]:
        self._record(el, rule, family=el.family)
        return [self._drift_card(el, self._z(p), kind="Instrument")]

    def _as_drift(self, el: Element, p, rule) -> list[_Card]:
        self._record(el, rule, length=el.length)
        return [self._drift_card(el, self._z(p), kind=el.kind)]

    def _w_foil(self, el: Foil, p, rule) -> list[_Card]:
        return self._as_drift(el, p, rule)

    def _w_taylor(self, el: Taylor, p, rule) -> list[_Card]:
        return self._as_drift(el, p, rule)

    def _w_patch(self, el: Patch, p, rule) -> list[_Card]:
        return self._as_drift(el, p, rule)

    def _w_referencechange(self, el, p, rule) -> list[_Card]:
        self._record(el, rule)
        return []

    def _w_superposition(self, el: Superposition, p: Placed, rule) -> list[_Card]:
        self._record(el, rule)
        cards: list[_Card] = []
        s = self._z(p)
        for _off, name in el.children:
            child = self.lat.elements.get(name)
            if child is None:
                continue
            sub = p.model_copy(update={"element": child, "s_in": s, "s_out": s + child.length})
            fn = getattr(self, f"_w_{child.kind.lower()}")
            for c in fn(child, sub, self.RULES[child.kind]):
                c.name, c.kind = c.name or child.name, c.kind or child.kind
                cards.append(c)
            s += child.length
        return cards

    # ------------------------------------------------------------------ geometry fix-ups
    def _absorb_synthetic(self, cards: list[_Card]) -> None:
        """A thin element given a length overlaps its neighbours: take half of that length out of
        the drift card before it and half out of the one after (Σ length preserved, PLAN I-1); what
        no drift can give up comes from free space between cards, and only then from moving
        everything downstream."""
        for i, c in enumerate(cards):
            if not c.synthetic:
                continue
            before = _nearest_drift(cards, i, -1)
            after = _nearest_drift(cards, i, +1)
            if c.rf is not None and c.rf.get("adaptive"):
                space = (cards[before].length if before is not None else 0.0) \
                    + (cards[after].length if after is not None else 0.0)
                c.length = round(max(_THIN_SURROGATE_MIN, min(c.rf["wanted"], space)), 12)
                self.rep.equivalent("THIN_GAP_AS_SHORT_CAVITY",
                                    f"IMPACT-T has no thin gap; written as a {c.length:.6g} m type-104 cavity "
                                    "(a half-wave bump, or what the neighbouring drifts can give up) centred at "
                                    "the gap", element=c.name, kind=c.kind, length_m=c.length)
            elif c.rf is not None:
                self.rep.equivalent("THIN_GAP_AS_SHORT_CAVITY",
                                    f"IMPACT-T has no thin gap; written as a {c.length:.6g} m type-104 cavity "
                                    "centred at the gap, its length taken from the neighbouring drifts",
                                    element=c.name, kind=c.kind, length_m=c.length)
            half = 0.5 * c.length
            take_b = min(half, cards[before].length) if before is not None else 0.0
            take_a = min(c.length - take_b, cards[after].length) if after is not None else 0.0
            if take_b + take_a < c.length - 1e-15 and before is not None:
                take_b = min(c.length - take_a, cards[before].length)
            # the nominal position: exact for a gap (its tag ``pad`` must add up to it on the reader's
            # picometre grid), the card's own centre for a thin multipole
            centre = c.rf["centre"] if c.rf is not None else c.zedge + half
            take_b, take_a = round(take_b, 12), round(take_a, 12)        # a picometre grid keeps rewrites exact
            if before is not None and take_b:
                cards[before].length = round(cards[before].length - take_b, 12)
            if after is not None and take_a:
                cards[after].length = round(cards[after].length - take_a, 12)
                cards[after].values[0] = round(cards[after].zedge + take_a, 12)
            c.values[0] = round(centre - take_b, 12)   # the element spans exactly what the drifts gave up
            nxt = next((o.zedge for o in cards[i + 1:] if o.length > 0 or o.itype == -99), None)
            shift = max(0.0, c.zedge + c.length - nxt) if nxt is not None else 0.0
            if shift > 1e-9:
                for other in cards[i + 1:]:            # keep the sequence contiguous: everything downstream moves
                    _shift_card(other, shift)
                self.rep.lossy("THIN_GAP_ADDS_LENGTH",
                               f"the {c.length:.6g} m element standing in for a thin one could not be absorbed by "
                               f"neighbouring drifts; the lattice grows by {shift:.6g} m",
                               element=c.name, kind=c.kind, added_length_m=shift)
            # the tag lets the reader put the thin element back where it was and return the length
            # to the drifts that gave it up (``card_span``)
            c.tag_extra = f"L=0 pad={fmt(take_b)}"
            # a reader restores the thin element and returns the absorbed length to the drifts; the
            # added length stays as a drift after it, crossed at the exit velocity: keep the driven
            # phases downstream consistent with that time of flight
            if shift > 1e-9 and c.beta_out > 0:
                dt = shift / (c.beta_out * C_LIGHT)
                for o in cards[i + 1:]:
                    if o.rf is not None:
                        o.rf["t0"] += dt
                        o.rf["s0"] += shift


def _nearest_drift(cards: list[_Card], idx: int, step: int) -> int | None:
    j = idx + step
    while 0 <= j < len(cards):
        c = cards[j]
        if c.itype == 0 and c.length > 0:
            return j
        if c.length > 0 or c.itype in (-99,):
            return None
        j += step
    return None


def _has_shift(el: Element) -> bool:
    s = el.shift
    if s is not None and not s.is_zero():
        return True
    mp = getattr(el, "multipole", None)
    return bool(mp is not None and any(mp.tilt.values()))


def write(lattice: Lattice, path, **options) -> FidelityReport:
    return Writer().write(lattice, path, **options)


assert check_rules_coverage(Writer()) == set()
