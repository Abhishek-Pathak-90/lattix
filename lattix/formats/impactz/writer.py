"""IMPACT-Z ``ImpactZ.in`` writer (PLAN §6 task 3.3).

The deck is one file: eleven header records built from ``lattice.reference`` and the
writer's options, then one card per element, then a ``-99`` terminator.  Numbers are
written with ``%.15g`` (PLAN Phase-1 measurement: ``%.10g`` already costs 4e-8 on a
converted gradient).  The column meanings and their evidence in the IMPACT-Z sources
are tabulated in :mod:`lattix.formats.impactz.reader`.

Choices this writer makes, and why
----------------------------------
* **Names.** IMPACT-Z cards are unnamed.  Every card is preceded by a
  ``! lattix: name=<percent-encoded> kind=<IR kind>`` comment which
  :class:`~lattix.formats.impactz.reader.Reader` reads back, so names survive a round
  trip (PLAN §4.4, invariant I-15).  IMPACT-Z itself skips any line whose first
  list-directed item is ``!`` (``Input.f90:69``).
* **Integrator.** ``integrator="map"`` (header ``flagmap = 1``, the default) makes
  IMPACT-Z use its linear transfer maps — MEASURED 2026-09-03 to reproduce the
  analytic thick-quadrupole map to 4e-14 with ``map_steps >= 20``.
  ``integrator="lorentz"`` (``flagmap = 2``) switches to the nonlinear Lorentz
  integrator, which is the only one that evaluates the real multipole field.
  ``intfunc1_Multipole`` (``Multipole.f90:269-285``) tests ``Param(3)`` — the *field
  strength* — against 1e-5 and, when it is larger, interpolates ``zdat``/``edat``,
  which nothing ever loads for a type-5 element (``AccSimulator.f90`` calls
  ``read1_Data`` only for types 1 and > 100); when it is smaller it uses ``Param(2)``,
  the *order id*, as a quadrupole gradient.  MEASURED: a 0.2 m sextupole under
  ``flagmap = 1`` produces a NaN map.  Writing a Sextupole/Octupole with the map
  integrator therefore records ``LOSSY:IMPACTZ_MULTIPOLE_LINEAR_MAP``.
* **The negative-``Param(5)`` sentinel.** ``drift1_BeamBunch`` (``BeamBunch.f90:323``)
  treats *any* element other than types 0, 1 and 4 whose ``Param(5)`` is negative as an
  ideal RF cavity.  For a solenoid ``Param(5)`` is the **x misalignment**, so a negative
  ``dx`` silently turns it into an accelerating gap (MEASURED: the 4×4 map becomes a
  drift and the reference energy moves).  The writer refuses to emit that: a negative
  solenoid ``dx`` is zeroed and recorded ``LOSSY:IMPACTZ_SOLENOID_DX_SENTINEL``.
* **RF.** IMPACT-Z has no thin gap, but it does have an *ideal cavity* model that is
  exactly the IR's: when the file-ID column of an element of type > 100 is **negative**
  (``drift1_BeamBunch``, ``BeamBunch.f90:323-437``, reached for every type but 0, 1 and
  4 under ``flagmap = 1``), IMPACT-Z ignores the field file and integrates
  ``dγ/dz = (E_0/mc²)·cos(φ_s)`` for the reference and ``cos(Δφ·h + φ_s)`` for each
  particle, with entrance/exit radial impulses.  ``Param(2)`` is then the accelerating
  **gradient in V/m**, ``Param(4)`` a **synchronous phase in degrees** measured from the
  crest, and the reference gain is ``E_0·L·cos φ_s`` *independently of the charge* —
  the IR's rule verbatim.  MEASURED 2026-09-03 on a 1 MV gap at φ_s = −30°:
  ΔE = 866 025.403 784 337 eV against the IR's 866 025.403 784 439 (1.2e-16 relative),
  ``det(2×2) = p_in/p_out`` to 5e-12 and the common-basis ``R65 = −4.3046``, which is
  the value ``docs/oracles.md`` pins for Bmad's ``lcavity`` and HELIX.
  ``rf_model="rfdata"`` instead writes a generated ``rfdataN.in`` holding a raised-cosine
  on-axis profile, with the amplitude and phase corrected by the complex form factor
  ``∫Ez e^{ik(z−z_c)}dz / ∫Ez dz`` (k = ω/βc); that path needs the absolute time of
  flight, because ``Param(4)`` is a *driven* RF phase there (``SC.f90:143-176``).
* A thin cavity still needs a length: the gap length is taken out of the neighbouring
  drifts (half from each) so ``Σ length`` is preserved, and ``LOSSY:THIN_GAP_ADDS_LENGTH``
  is recorded when there is not enough drift to take it from.
* **Bends** are Transport-map dipoles (type 4).  Angles and pole-face angles are in
  **radians** (the ``*pi/180`` conversions in ``AccSimulator.f90:1049-1055`` are
  commented out), ``Param(5)`` is the half gap and ``Param(10)`` is MAD's ``fint``.
  A vertical bend (``tilt_ref = ±π/2``) has no IMPACT-Z representation.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from lattix import __version__
from lattix.fidelity import FidelityReport
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
    ReferenceChange,
    RFCavity,
    RFQCell,
    Solenoid,
    Superposition,
    Taylor,
)
from lattix.ir.lattice import Lattice, Placed
from lattix.ir.reference import ReferenceParticle
from lattix.ir.units import C_LIGHT
from lattix.ir.walk import energy_gain_eV, propagate

_TOL = 1e-15


def fmt(x: float) -> str:
    """``%.15g`` with ``-0`` normalised away (Fortran list-directed input reads it)."""
    if x == 0:
        x = 0.0
    return f"{x:.15g}"


@dataclass(frozen=True)
class Rule:
    """One row of the writer's capability matrix (PLAN §4.4)."""

    target: str
    cls: str = "EXACT"
    code: str = "OK"
    message: str = ""


@dataclass
class _Card:
    """A pending element line."""

    length: float
    nseg: int
    mapstp: int
    itype: int
    values: list[float]
    name: str = ""
    kind: str = ""
    note: str = ""

    def render(self) -> str:
        vals = " ".join(fmt(v) for v in self.values)
        head = f"{fmt(self.length)} {self.nseg:d} {self.mapstp:d} {self.itype:d}"
        line = f"{head} {vals} /" if vals else f"{head} /"
        return f"{line} {self.note}" if self.note else line


class Writer:
    """``Writer().write(lattice, path)`` → :class:`FidelityReport`."""

    format = "impactz"

    RULES: dict[str, Rule] = {
        "Drift": Rule("0 drift tube"),
        "Quadrupole": Rule("1 quadrupole (lab gradient T/m)"),
        "Sextupole": Rule("5 multipole id 2 (B'' T/m²)"),
        "Octupole": Rule("5 multipole id 3 (B''' T/m³)"),
        "Multipole": Rule("-55 thin-lens multipole (k0…k5)", "EQUIVALENT",
                          "THIN_MULTIPOLE_AS_KICK",
                          "thin multipole written as the -55 thin-lens kick (normal "
                          "components only, k_n = B_nL/Bρ)"),
        "Bend": Rule("4 dipole (Transport map + pole faces)"),
        "Solenoid": Rule("3 solenoid (Bz0 T)"),
        "RFCavity": Rule("104 ideal cavity (negative file ID)", "EQUIVALENT",
                         "THIN_GAP_AS_SHORT_CAVITY",
                         "IMPACT-Z has no thin gap; written as a short type-104 cavity "
                         "running its ideal-cavity model (dE = E0·L·cos φs exactly)"),
        "FieldMap": Rule("104 SC cavity + rfdataN.in", "EQUIVALENT", "FM_AS_RFDATA",
                         "field map written as a type-104 cavity whose rfdata file holds "
                         "the on-axis Ez profile"),
        "NCells": Rule("103 CCL ideal cavity", "EQUIVALENT", "NCELLS_AS_CCL",
                       "cell train written as a type-103 coupled-cavity structure driven "
                       "by IMPACT-Z's ideal-cavity model"),
        "RFQCell": Rule("0 drift", "LOSSY", "RFQ_TO_DRIFT",
                        "IMPACT-Z has no RFQ cell; written as a drift of the same length"),
        "Kicker": Rule("-21 centroid shift (dpx, dpy in rad)"),
        "Collimator": Rule("-13 collimator slit"),
        "Marker": Rule("0 drift of zero length"),
        "Instrument": Rule("0 drift of zero length", "LOSSY", "INSTRUMENT_TO_MARKER",
                           "IMPACT-Z diagnostics write numbered fort files; the monitor is "
                           "written as a zero-length drift instead"),
        "Foil": Rule("0 drift", "DROPPED", "FOIL_DROPPED",
                     "IMPACT-Z has no stripping foil; written as a drift"),
        "Taylor": Rule("0 drift", "DROPPED", "TAYLOR_DROPPED",
                       "IMPACT-Z's -12 external map reads a file this writer does not "
                       "generate; written as a drift"),
        "Patch": Rule("0 drift", "LOSSY", "PATCH_DROPPED",
                      "IMPACT-Z has no patch element; written as a drift"),
        "ReferenceChange": Rule("(nothing)", "LOSSY", "REFCHANGE_DROPPED",
                                "IMPACT-Z's reference energy follows its own integration "
                                "and cannot be reset by an element"),
        "Freq": Rule("(nothing)", "EXACT", "OK",
                     "the RF clock lives on each cavity's frequency column"),
        "Directive": Rule("comment", "DROPPED", "FOREIGN_DIRECTIVE",
                          "format-specific directive written as a comment only"),
        "Superposition": Rule("0 drift", "LOSSY", "SUPERPOSITION_TO_DRIFT",
                              "overlapping field maps have no IMPACT-Z element; written as "
                              "a drift of the same length"),
    }

    #: ``Directive.role``s that lose nothing when written as a plain comment.
    COMMENT_ROLES = frozenset({"period_start", "period_end", "sync_phase", "title"})

    # ------------------------------------------------------------------
    def write(self, lattice: Lattice, path: Path, *, strict: bool = False,
              n_particles: int = 1000, grid: tuple[int, int, int] = (64, 64, 64),
              steps_per_m: float = 10.0, map_steps: int = 20, integrator: str = "map",
              current_A: float = 0.0, distribution: int = 3,
              radius_m: float = 0.014, perdlen_m: float = 0.1, flagbc: int = 1,
              flagdiag: int = 1, twiss: dict | None = None, errors: bool | None = None,
              thin_gap_length_m: float = 1e-3, rfdata_points: int = 65,
              first_rfdata_id: int = 1, phase_ini_rad: float = 0.0, rf_model: str = "ideal",
              use: str | None = None, write_rfdata: bool = True) -> FidelityReport:
        """Write *lattice* as an ``ImpactZ.in`` deck.

        ``steps_per_m`` sets each element's ``bnseg`` (space-charge / diagnostic
        steps, at least 1), ``map_steps`` its ``bmpstp`` (map integration steps;
        20 reproduces the analytic thick-quad map to 4e-14).  ``integrator`` is
        ``"map"`` (``flagmap = 1``) or ``"lorentz"`` (``flagmap = 2``).
        ``distribution`` is the header ``flagdist`` (3 = waterbag, 2 = Gaussian,
        19 = read ``particle.in``).  ``rf_model`` is ``"ideal"`` (IMPACT-Z's built-in
        ideal cavity, exact for the IR's ``dE = V·cos φ_s``) or ``"rfdata"`` (a
        generated on-axis profile).  Generated ``rfdataN.in`` files are written next
        to *path* unless ``write_rfdata=False``.
        """
        if integrator not in ("map", "lorentz"):
            raise ValueError(f"integrator must be 'map' or 'lorentz', got {integrator!r}")
        if rf_model not in ("ideal", "rfdata"):
            raise ValueError(f"rf_model must be 'ideal' or 'rfdata', got {rf_model!r}")
        if map_steps < 1 or steps_per_m <= 0:
            raise ValueError("map_steps must be >= 1 and steps_per_m > 0")
        path = Path(path)
        rep = FidelityReport(target_format="impactz", target_file=str(path))
        self.rep = rep
        self.lat = lattice
        self.opts = dict(steps_per_m=steps_per_m, map_steps=map_steps,
                         radius_m=radius_m, thin_gap_length_m=thin_gap_length_m,
                         rfdata_points=rfdata_points, integrator=integrator,
                         rf_model=rf_model)
        self.flagmap = 1 if integrator == "map" else 2
        self.rfdata: dict[int, str] = {}          # file id -> text
        self._rfdata_by_hash: dict[str, int] = {}
        self._pending_gaps: list[tuple[str, float]] = []
        self._next_rfdata = int(first_rfdata_id)

        placed = propagate(lattice, use)
        self.errors = (any(_needs_errors(p.element) for p in placed)
                       if errors is None else bool(errors))
        cards: list[_Card] = []
        comments: list[tuple[int, str]] = []      # (index into cards, comment text)
        for i, p in enumerate(placed):
            for c in self._element(p, placed, i, comments, len(cards)):
                cards.append(c)
        self._absorb_gap_lengths(placed, cards, rep)
        cards.append(_Card(0.0, 0, 0, -99, [], name="", kind="", note="! end"))

        text = self._render(lattice, cards, comments, placed,
                            n_particles=n_particles, grid=grid, current_A=current_A,
                            distribution=distribution, radius_m=radius_m,
                            perdlen_m=perdlen_m, flagbc=flagbc, flagdiag=flagdiag,
                            twiss=twiss, phase_ini_rad=phase_ini_rad)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if write_rfdata:
            for fid, body in sorted(self.rfdata.items()):
                (path.parent / f"rfdata{fid}.in").write_text(body)
        rep.raise_if(strict)
        return rep

    # ------------------------------------------------------------------
    # header
    # ------------------------------------------------------------------
    def _render(self, lat: Lattice, cards: list[_Card], comments: list[tuple[int, str]],
                placed: list[Placed], *, n_particles: int, grid, current_A: float,
                distribution: int, radius_m: float, perdlen_m: float, flagbc: int,
                flagdiag: int, twiss: dict | None, phase_ini_rad: float) -> str:
        ref = lat.reference
        freq = self._frequency(lat, placed)
        sp = ref.species
        qmcc = sp.charge / sp.mass_eV
        tw = {"alfx": 0.0, "betx": 1.0, "emitx": 1e-6, "alfy": 0.0, "bety": 1.0,
              "emity": 1e-6, "alfz": 0.0, "betz": 1.0, "emitz": 1e-3}
        tw.update(twiss or {})
        nx, ny, nz = (int(v) for v in grid)
        out = [
            f"! IMPACT-Z deck written by lattix {__version__} from "
            f"{lat.meta.get('source_format', 'the lattix IR')}",
            f"! lattice: {lat.name!r} — {len(placed)} elements, "
            f"{sum(p.length for p in placed):.15g} m",
            "! header: 11 records (comment lines starting with ! are skipped)",
            "! 1: npcol nprow",
            "1 1",
            "! 2: dim np flagmap(1=linear map,2=Lorentz) flagerr flagdiag",
            f"6 {int(n_particles)} {self.flagmap} {1 if self.errors else 0} {int(flagdiag)}",
            "! 3: nx ny nz flagbc xrad[m] yrad[m] perdlen[m]",
            f"{nx} {ny} {nz} {int(flagbc)} {fmt(radius_m)} {fmt(radius_m)} {fmt(perdlen_m)}",
            "! 4: flagdist rstartflg flagsbstp nchrg",
            f"{int(distribution)} 0 0 1",
            "! 5: # of particles per charge state",
            f"{int(n_particles)}",
            "! 6: current per charge state [A]",
            f"{fmt(current_A)}",
            "! 7: q/m per charge state [1/eV]",
            f"{fmt(qmcc)}",
            "! 8: x  alpha beta[m] emit_n[m-rad] xscale pxscale xmu1 xmu2",
            f"{fmt(tw['alfx'])} {fmt(tw['betx'])} {fmt(tw['emitx'])} 1 1 0 0",
            "! 9: y  alpha beta[m] emit_n[m-rad] yscale pyscale xmu3 xmu4",
            f"{fmt(tw['alfy'])} {fmt(tw['bety'])} {fmt(tw['emity'])} 1 1 0 0",
            "! 10: z alpha beta[deg/MeV] emit[deg-MeV] zscale pzscale xmu5 xmu6",
            f"{fmt(tw['alfz'])} {fmt(tw['betz'])} {fmt(tw['emitz'])} 1 1 0 0",
            "! 11: current[A] kinetic_energy[eV] mass[eV] charge[e] frequency[Hz] phase[rad]",
            f"{fmt(current_A)} {fmt(ref.kinetic_energy_eV)} {fmt(sp.mass_eV)} "
            f"{fmt(float(sp.charge))} {fmt(freq)} {fmt(phase_ini_rad)}",
            "!",
            "! lattice: length nseg mapstp type value1 … / (a card ends at the '/')",
        ]
        by_index: dict[int, list[str]] = {}
        for idx, txt in comments:
            by_index.setdefault(idx, []).append(txt)
        for i, c in enumerate(cards):
            for txt in by_index.get(i, ()):
                out.append(txt)
            if c.name:
                out.append(f"! lattix: name={quote(c.name, safe='')} kind={c.kind}")
            out.append(c.render())
        return "\n".join(out) + "\n"

    def _frequency(self, lat: Lattice, placed: list[Placed]) -> float:
        """Header reference frequency: the RF clock, else the first RF element's."""
        if lat.reference.rf_frequency_Hz:
            return float(lat.reference.rf_frequency_Hz)
        for p in placed:
            if isinstance(p.element, Freq) and p.element.frequency_Hz:
                return float(p.element.frequency_Hz)
            rf = getattr(p.element, "rf", None)
            if rf is not None and rf.frequency_Hz:
                return float(rf.frequency_Hz)
        self.rep.lossy("IMPACTZ_NO_FREQUENCY",
                       "no RF frequency anywhere in the lattice; the header frequency "
                       "(which sets IMPACT-Z's internal length scale c/2πf) defaults to "
                       "1 GHz")
        return 1e9

    # ------------------------------------------------------------------
    # per-element dispatch
    # ------------------------------------------------------------------
    def _nseg(self, length: float) -> int:
        return max(1, int(math.ceil(abs(length) * self.opts["steps_per_m"] - 1e-12)))

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
            self.rep.lossy("IMPACTZ_MISALIGNMENT_DROPPED",
                           "the deck's flagerr is 0, so IMPACT-Z would ignore the "
                           "misalignment columns; they are written as zeros "
                           "(pass errors=True to apply them)",
                           element=el.name, kind=el.kind)
            return [0.0] * 5
        if s.z_offset:
            self.rep.lossy("IMPACTZ_NO_Z_OFFSET",
                           "IMPACT-Z misalignments have no longitudinal offset",
                           element=el.name, kind=el.kind, z_offset=s.z_offset)
        return [s.x_offset, s.y_offset, s.x_rot, s.y_rot, s.tilt]

    def _record(self, el: Element, rule: Rule, **details) -> None:
        if rule.cls == "EXACT":
            self.rep.exact(el.name, el.kind, code=rule.code, message=rule.message)
        else:
            self.rep.add(rule.cls, rule.code, rule.message, element=el.name,
                         kind=el.kind, **details)

    def _element(self, p: Placed, placed: list[Placed], i: int,
                 comments: list[tuple[int, str]], card_index: int) -> list[_Card]:
        el = p.element
        rule = self.RULES.get(el.kind)
        if rule is None:                                   # pragma: no cover - RULES is total
            raise KeyError(f"the IMPACT-Z writer has no rule for kind {el.kind!r}")
        if isinstance(el, Freq):
            self._record(el, rule)
            return []
        if isinstance(el, Directive):
            self._record(el, rule, role=el.role, card=el.card)
            comments.append((card_index,
                             f"! lattix: dropped {el.format} directive "
                             f"{el.card} {' '.join(el.args)}".rstrip()))
            return []
        native = el.native.get("impactz")
        if native and native.get("passthrough"):
            self.rep.exact(el.name, el.kind, code="IMPACTZ_NATIVE_PASSTHROUGH",
                           message=f"type {native['type']} re-emitted from native['impactz']")
            return [_Card(el.length, int(native["nseg"]), int(native["mapstp"]),
                          int(native["type"]), list(native["values"]),
                          name=el.name, kind=el.kind)]
        fn = getattr(self, f"_w_{el.kind.lower()}")
        cards = fn(el, p, rule)
        for c in cards:
            c.name = c.name or el.name
            c.kind = c.kind or el.kind
        return cards

    # -- magnets ----------------------------------------------------------
    def _w_drift(self, el, p, rule) -> list[_Card]:
        self._record(el, rule)
        return [self._drift_card(el)]

    def _drift_card(self, el: Element, length: float | None = None,
                    kind: str | None = None) -> _Card:
        """A type-0 card.  ``kind`` is the tag the reader will see: the element's own
        kind when :attr:`Reader.PROMOTIONS` can restore it, ``"Drift"`` otherwise (a
        conditional degradation such as ``FM_TO_DRIFT`` must not claim to be a field
        map on the way back)."""
        L = el.length if length is None else length
        return _Card(L, self._nseg(L), self.opts["map_steps"], 0, [self._radius(el)],
                     kind=kind or "Drift")

    def _w_quadrupole(self, el, p, rule) -> list[_Card]:
        mp = el.multipole
        if mp.tilt.get(1):
            self.rep.lossy("IMPACTZ_SKEW_QUAD",
                           "IMPACT-Z type 1 has no roll parameter separate from the "
                           "rotation-error column; the skew angle is written as rot_z, "
                           "which IMPACT-Z applies only when the header flagerr is 1",
                           element=el.name, kind="Quadrupole", tilt=mp.tilt[1])
        extra = [v for k, v in sorted(mp.Bn.items()) if k != 1 and v]
        if extra or any(mp.Bs.values()):
            self.rep.lossy("IMPACTZ_QUAD_HIGHER_ORDER",
                           "higher-order components inside a quadrupole are dropped "
                           "(IMPACT-Z type 1 carries only the gradient)",
                           element=el.name, kind="Quadrupole")
        self._record(el, rule)
        sh = self._shift_cols(el)
        sh[4] += mp.tilt.get(1, 0.0)
        return [_Card(el.length, self._nseg(el.length), self.opts["map_steps"], 1,
                      [mp.Bn.get(1, 0.0), 0.0, self._radius(el), *sh])]

    def _w_sextupole(self, el, p, rule) -> list[_Card]:
        return self._multipole_card(el, rule, order=2, ident=2)

    def _w_octupole(self, el, p, rule) -> list[_Card]:
        return self._multipole_card(el, rule, order=3, ident=3)

    def _multipole_card(self, el, rule, *, order: int, ident: int) -> list[_Card]:
        self._record(el, rule)
        if self.flagmap == 1:
            self.rep.lossy("IMPACTZ_MULTIPOLE_LINEAR_MAP",
                           "IMPACT-Z's linear-map integrator reads Param(2) (the "
                           "multipole order id) as the gradient for type 5 "
                           "(Multipole.f90:269-278), so this element only behaves as a "
                           f"{el.kind.lower()} with integrator='lorentz'",
                           element=el.name, kind=el.kind)
        if el.multipole.Bs.get(order):
            self.rep.lossy("IMPACTZ_SKEW_MULTIPOLE",
                           "IMPACT-Z type 5 has no skew component",
                           element=el.name, kind=el.kind)
        return [_Card(el.length, self._nseg(el.length), self.opts["map_steps"], 5,
                      [float(ident), el.multipole.Bn.get(order, 0.0), 0.0,
                       self._radius(el), *self._shift_cols(el)])]

    def _w_multipole(self, el: Multipole, p: Placed, rule) -> list[_Card]:
        """Thin multipole → ``-55`` (``thinMultipole_BPM``, ``BPM.f90:452-489``):
        ``Δp_x = −Σ k_n/n!·Re[(x+iy)^n]``, i.e. ``k_n ≡ MAD's knl``."""
        brho = p.ref_in.brho_signed if p.ref_in else self.lat.reference.brho_signed
        mp = el.multipole
        ks = [mp.BnL.get(n, 0.0) / brho for n in range(6)]
        if any(mp.BsL.values()):
            self.rep.lossy("IMPACTZ_SKEW_MULTIPOLE",
                           "IMPACT-Z's -55 thin multipole has normal components only",
                           element=el.name, kind="Multipole")
        if any(n > 5 for n, v in mp.BnL.items() if v):
            self.rep.lossy("IMPACTZ_MULTIPOLE_ORDER",
                           "IMPACT-Z's -55 thin multipole stops at k5",
                           element=el.name, kind="Multipole")
        self._record(el, rule)
        cards = [_Card(0.0, 0, 0, -55, [0.0, *ks])]
        if el.length:
            cards.append(self._drift_card(el))
            self.rep.lossy("IMPACTZ_THICK_MULTIPOLE_SPLIT",
                           "a thin multipole with a length is written as the -55 kick "
                           "followed by a drift", element=el.name, kind="Multipole")
        return cards

    def _w_bend(self, el: Bend, p: Placed, rule) -> list[_Card]:
        b = el.bend
        if abs(b.tilt_ref) > 1e-12:
            self.rep.lossy("IMPACTZ_NO_REF_TILT",
                           "IMPACT-Z's type-4 dipole bends in the horizontal plane only; "
                           "the reference tilt is dropped",
                           element=el.name, kind="Bend", tilt_ref=b.tilt_ref)
        brho = p.ref_in.brho_signed if p.ref_in else self.lat.reference.brho_signed
        k1 = el.multipole.Bn.get(1, 0.0) / brho
        e1, e2 = b.e1, b.e2
        if b.rect:
            # the IR stores rectangular e1/e2 already including angle/2 (PLAN §4.3)
            pass
        fint2 = b.edge_int2
        if fint2 is not None and abs(fint2 - b.edge_int1) > 1e-12:
            self.rep.lossy("IMPACTZ_SINGLE_FINT",
                           "AccSimulator.f90:1057 sets the exit fringe integral equal to "
                           "the entrance one (Kb = Kf); fintx is dropped",
                           element=el.name, kind="Bend", fint=b.edge_int1, fintx=fint2)
        if b.fringe_k2 is not None:
            self.rep.lossy("IMPACTZ_NO_FRINGE_K2",
                           "IMPACT-Z's Transport dipole has no second fringe coefficient",
                           element=el.name, kind="Bend", K2=b.fringe_k2)
        self._record(el, rule)
        # file ID > 100 selects the z-map Transport integration (AccSimulator.f90:1027)
        return [_Card(el.length, self._nseg(el.length), self.opts["map_steps"], 4,
                      [b.angle, k1, 150.0, b.hgap, e1, e2, 0.0, 0.0, b.edge_int1,
                       *self._shift_cols(el)])]

    def _w_solenoid(self, el: Solenoid, p, rule) -> list[_Card]:
        self._record(el, rule)
        sh = self._shift_cols(el)
        if sh[0] < 0.0:
            # Param(5) of a solenoid is dx, and drift1_BeamBunch treats a negative
            # Param(5) as the ideal-cavity sentinel (BeamBunch.f90:323) — MEASURED to
            # turn the solenoid into an accelerating gap.
            self.rep.lossy("IMPACTZ_SOLENOID_DX_SENTINEL",
                           f"a negative solenoid x offset ({sh[0]:.15g} m) would land in "
                           "IMPACT-Z's Param(5) < 0 ideal-cavity sentinel; written as 0",
                           element=el.name, kind="Solenoid", x_offset=sh[0])
            sh[0] = 0.0
        return [_Card(el.length, self._nseg(el.length), self.opts["map_steps"], 3,
                      [el.solenoid.Bsol_T, 0.0, self._radius(el), *sh])]

    # -- diagnostics / apertures ------------------------------------------
    def _w_kicker(self, el: Kicker, p, rule) -> list[_Card]:
        if el.electric:
            self.rep.lossy("IMPACTZ_ELECTRIC_KICKER",
                           "IMPACT-Z's -21 shift is species-independent; the electric "
                           "flag is dropped", element=el.name, kind="Kicker")
        self._record(el, rule)
        cards = [_Card(0.0, 0, 0, -21, [0.0, 0.0, el.hkick, 0.0, el.vkick, 0.0, 0.0])]
        if el.length:
            cards.append(self._drift_card(el))
            self.rep.lossy("IMPACTZ_THICK_KICKER_SPLIT",
                           "a kicker with a length is written as the -21 kick followed "
                           "by a drift", element=el.name, kind="Kicker")
        return cards

    def _w_collimator(self, el: Collimator, p, rule) -> list[_Card]:
        ap = el.aperture
        r = self._radius(el)
        xl = ap.x_limits if ap and ap.x_limits else (-r, r)
        yl = ap.y_limits if ap and ap.y_limits else (-r, r)
        if ap is not None and ap.shape == "ELLIPTICAL":
            self.rep.lossy("IMPACTZ_ELLIPSE_AS_SLIT",
                           "IMPACT-Z's -13 collimator is a rectangular slit; an "
                           "elliptical aperture becomes its bounding box",
                           element=el.name, kind="Collimator")
        else:
            self._record(el, rule)
        cards = [_Card(0.0, 0, 0, -13, [0.0, xl[0], xl[1], yl[0], yl[1]])]
        if el.length:
            cards.append(self._drift_card(el))
        return cards

    def _w_marker(self, el: Marker, p, rule) -> list[_Card]:
        self._record(el, rule)
        return [self._drift_card(el, 0.0, kind="Marker")]

    def _w_instrument(self, el: Instrument, p, rule) -> list[_Card]:
        self._record(el, rule, family=el.family)
        return [self._drift_card(el, kind="Instrument")]

    # -- degradations ------------------------------------------------------
    def _as_drift(self, el: Element, rule) -> list[_Card]:
        self._record(el, rule, length=el.length)
        return [self._drift_card(el, kind=el.kind)]

    def _w_foil(self, el: Foil, p, rule) -> list[_Card]:
        return self._as_drift(el, rule)

    def _w_taylor(self, el: Taylor, p, rule) -> list[_Card]:
        return self._as_drift(el, rule)

    def _w_patch(self, el: Patch, p, rule) -> list[_Card]:
        return self._as_drift(el, rule)

    def _w_rfqcell(self, el: RFQCell, p, rule) -> list[_Card]:
        return self._as_drift(el, rule)

    def _w_superposition(self, el: Superposition, p, rule) -> list[_Card]:
        return self._as_drift(el, rule)

    def _w_referencechange(self, el: ReferenceChange, p, rule) -> list[_Card]:
        dE = energy_gain_eV(el, p.ref_in or self.lat.reference)
        self._record(el, rule, dE_ref_eV=dE)
        return [self._drift_card(el, kind="ReferenceChange")]

    # ------------------------------------------------------------------
    # RF
    # ------------------------------------------------------------------
    def _w_rfcavity(self, el: RFCavity, p: Placed, rule) -> list[_Card]:
        ref = p.ref_in or self.lat.reference
        rf = el.rf
        V = rf.voltage_V
        if not V and rf.gradient_V_per_m is not None:
            V = rf.gradient_V_per_m * (rf.L_active_m if rf.L_active_m is not None
                                       else el.length)
        if rf.dE_ref_eV is not None and rf.dE_ref_eV and abs(math.cos(rf.phase_rad)) > 1e-12:
            V = rf.dE_ref_eV / math.cos(rf.phase_rad)
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        if not freq or not V:
            self.rep.lossy("IMPACTZ_CAVITY_TO_DRIFT",
                           "an RF cavity without a frequency or a voltage cannot be "
                           "written; replaced by a drift",
                           element=el.name, kind="RFCavity", voltage_V=V, frequency_Hz=freq)
            return [self._drift_card(el)]
        L_elem = el.length
        if L_elem <= 0:
            L_elem = float(self.opts["thin_gap_length_m"])
            self._pending_gaps.append((el.name, L_elem))
        # A thin gap keeps its position: half the synthetic length is taken from the
        # drift before and half from the drift after, so the *centre* of the generated
        # profile sits exactly where the IR put the gap (t_center = the walk's entry time).
        return self._rf_card(el, p, rule, itype=104, V=V, freq=freq, L_elem=L_elem,
                             L_active=rf.L_active_m or L_elem, phase=rf.phase_rad,
                             sync=rf.phase_is_sync, thin=el.length <= 0.0)

    def _rf_native_card(self, el: Element, p: Placed) -> list[_Card] | None:
        """Re-emit an RF card that references an ``rfdataN.in`` we did not generate.

        The IR owns the amplitude scale, the frequency, the driven phase, the file ID
        and the radius; every other column (misalignments, DTL quads, SolRF ``Bz0``)
        comes back from ``native['impactz']``.  The referenced ``rfdataN.in`` must be
        copied next to the new deck — recorded as ``EQUIVALENT:IMPACTZ_RFDATA_REFERENCE``.
        """
        native = el.native.get("impactz")
        fid = el.meta.get("impactz_file_id")
        if not native or fid is None or fid < 0:
            return None
        ref = p.ref_in or self.lat.reference
        rf = getattr(el, "rf", None)
        vals = list(native["values"]) + [0.0] * 5
        vals[0] = float(el.meta.get("impactz_scale", getattr(el, "ke", 1.0)))
        vals[1] = float((rf.frequency_Hz if rf else None) or ref.rf_frequency_Hz or 0.0)
        vals[2] = math.degrees(rf.phase_rad) if rf else 0.0
        vals[3] = float(fid)
        vals[4] = self._radius(el)
        del vals[len(native["values"]):]
        itype = int(el.meta.get("impactz_type", native["type"]))
        self.rep.equivalent("IMPACTZ_RFDATA_REFERENCE",
                            f"type {itype} re-emitted with its original rfdata{fid}.in "
                            "reference; copy that file next to the deck",
                            element=el.name, kind=el.kind, file=f"rfdata{fid}.in")
        return [_Card(el.length, self._nseg(el.length), self.opts["map_steps"], itype,
                      vals, name=el.name, kind=el.kind)]

    def _rf_card(self, el: Element, p: Placed, rule, *, itype: int, V: float, freq: float,
                 L_elem: float, L_active: float, phase: float, sync: bool,
                 shape=None, thin: bool = False) -> list[_Card]:
        """One accelerating card: IMPACT-Z's ideal cavity, or a generated profile."""
        ref = p.ref_in or self.lat.reference
        L_active = min(max(L_active, 0.0) or L_elem, L_elem)
        if not sync:
            self.rep.equivalent("IMPACTZ_RF_PHASE_NOT_SYNC",
                                "the source gave a driven RF phase, not a synchronous one; "
                                "written as IMPACT-Z's synchronous phase so the reference "
                                "gain still equals V·cos(phase)",
                                element=el.name, kind=el.kind)
        if self.opts["rf_model"] == "ideal":
            if self.flagmap != 1:
                self.rep.lossy("IMPACTZ_IDEAL_CAVITY_NEEDS_MAP",
                               "IMPACT-Z's ideal-cavity model lives in the linear-map "
                               "integrator only (BeamBunch.f90:323); with "
                               "integrator='lorentz' the negative file ID is meaningless",
                               element=el.name, kind=el.kind)
            phi_deg = (math.degrees(phase) + 180.0) % 360.0 - 180.0
            self._record(el, rule, voltage_V=V, length_m=L_elem, model="ideal",
                         gradient_V_per_m=V / L_elem, phase_deg=phi_deg)
            return [_Card(L_elem, self._nseg(L_elem), self.opts["map_steps"], itype,
                          [V / L_elem, freq, phi_deg, -1.0, self._radius(el),
                           *self._shift_cols(el)])]
        z, ez = shape if shape is not None else _raised_cosine(
            L_elem, L_active, self.opts["rfdata_points"])
        half = 0.0 if thin else 0.5 * (z[-1] - z[0])
        t_center = (ref.time_s or 0.0) + half / (ref.beta * C_LIGHT)
        theta0_deg, amp = self._rf_amplitude_phase(z, ez, V, phase, freq, ref, el, t_center)
        fid = self._rfdata_file(_with_derivative(z, [amp * v for v in ez]), derivative=True)
        self._record(el, rule, voltage_V=V, length_m=L_elem, model="rfdata",
                     rfdata=f"rfdata{fid}.in")
        self.rep.equivalent("IMPACTZ_RF_ABSOLUTE_PHASE",
                            "theta0 is a driven RF phase computed from the lattix "
                            "reference time of flight; IMPACT-Z re-integrates its own",
                            element=el.name, kind=el.kind, theta0_deg=theta0_deg)
        return [_Card(L_elem, self._nseg(L_elem), self.opts["map_steps"], itype,
                      [1.0, freq, theta0_deg, float(fid), self._radius(el),
                       *self._shift_cols(el)])]

    def _w_fieldmap(self, el: FieldMap, p: Placed, rule) -> list[_Card]:
        cards = self._rf_native_card(el, p)
        if cards is not None:
            self._record(el, rule, rfdata=f"rfdata{el.meta['impactz_file_id']}.in")
            return cards
        prof = self._map_profile(el)
        if prof is None:
            self.rep.lossy("FM_TO_DRIFT",
                           "no on-axis Ez profile available for this field map; written "
                           "as a drift of the same length",
                           element=el.name, kind="FieldMap", files=list(el.files))
            return [self._drift_card(el)]
        z, ez = prof
        ref = p.ref_in or self.lat.reference
        freq = el.rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        if not freq:
            self.rep.lossy("FM_TO_DRIFT", "field map without a frequency; written as a drift",
                           element=el.name, kind="FieldMap")
            return [self._drift_card(el)]
        phase = float(el.meta.get("map_summary", {}).get("phase_rf_rad", el.rf.phase_rad))
        # a field map's phase is the RF phase seen at the map ENTRANCE (TraceWin/PALS),
        # and the profile starts there, so no mid-element offset is applied
        theta0_deg = math.degrees(phase - 2.0 * math.pi * freq * (ref.time_s or 0.0))
        theta0_deg = (theta0_deg + 180.0) % 360.0 - 180.0
        rows = _with_derivative(z, ez)
        fid = self._rfdata_file(rows, derivative=True)
        self._record(el, rule, rfdata=f"rfdata{fid}.in", n_points=len(z))
        self.rep.equivalent("IMPACTZ_RF_ABSOLUTE_PHASE",
                            "theta0 is a driven RF phase computed from the lattix "
                            "reference time of flight; IMPACT-Z re-integrates the map",
                            element=el.name, kind=el.kind, theta0_deg=theta0_deg)
        return [_Card(el.length, self._nseg(el.length), self.opts["map_steps"], 104,
                      [1.0, freq, theta0_deg, float(fid), self._radius(el),
                       *self._shift_cols(el)])]

    def _w_ncells(self, el: NCells, p: Placed, rule) -> list[_Card]:
        ref = p.ref_in or self.lat.reference
        rf = el.rf
        V = rf.voltage_V or (rf.gradient_V_per_m or 0.0) * el.length
        freq = rf.frequency_Hz or (ref.rf_frequency_Hz or 0.0)
        n_cell = int(el.params.get("n_cells") or rf.n_cell or 1)
        cards = self._rf_native_card(el, p)
        if cards is not None:
            self._record(el, rule, rfdata=f"rfdata{el.meta['impactz_file_id']}.in")
            return cards
        if not (V and freq and el.length > 0):
            self.rep.lossy("NCELLS_TO_DRIFT",
                           "an NCells without a voltage, a frequency or a length is "
                           "written as a drift", element=el.name, kind="NCells")
            return [self._drift_card(el)]
        mode = el.params.get("mode")
        shape = (None if self.opts["rf_model"] == "ideal"
                 else _cell_train(el.length, n_cell, pi_mode=(mode != 0)))
        cards = self._rf_card(el, p, rule, itype=103, V=V, freq=freq, L_elem=el.length,
                              L_active=el.length, phase=rf.phase_rad,
                              sync=rf.phase_is_sync, shape=shape)
        unmapped = {k: v for k, v in el.params.items()
                    if k in ("beta_g", "k_eot_i", "k_eot_o", "dz_i", "dz_o", "p_flag") and v}
        unmapped.setdefault("n_cells", n_cell)
        unmapped.setdefault("mode", mode)
        self.rep.lossy("IMPACTZ_NCELLS_PARAMS",
                       "IMPACT-Z's type-103 structure has no cell count, no geometric β_g, "
                       "no entrance/exit half-cell corrections and no TTF tail; those "
                       "NCELLS columns are dropped",
                       element=el.name, kind="NCells", **unmapped)
        return cards

    # -- RF helpers --------------------------------------------------------
    def _rf_amplitude_phase(self, z, ez, V: float, phase_rad: float, freq: float,
                            ref: ReferenceParticle, el: Element,
                            t_center: float) -> tuple[float, float]:
        """Amplitude scale and ``theta0`` [deg] so that the reference gain is
        ``q·V·cos(phase_rad)``.

        IMPACT-Z integrates ``dγ/dz = (q/m)·Ez(z)·cos(2π f t(z) + θ0)``.  With
        ``t(z) ≈ t_in + z/(β c)`` the gain is ``Re[e^{iθ_c}·F]`` with
        ``F = ∫Ez e^{ik(z−z_c)}dz`` and ``k = 2π f/(β c)``; the amplitude is divided
        by ``|F|/∫Ez`` and the phase shifted by ``−arg F`` so the two agree.
        ``t_center`` is the reference time of flight at the middle of the profile.
        """
        beta = ref.beta
        k = 2.0 * math.pi * freq / (beta * C_LIGHT)
        zc = 0.5 * (z[0] + z[-1])
        num_r = num_i = den = 0.0
        for a, b in zip(range(len(z) - 1), range(1, len(z)), strict=True):
            dz = z[b] - z[a]
            for zz, e in ((z[a], ez[a]), (z[b], ez[b])):
                w = 0.5 * dz
                num_r += w * e * math.cos(k * (zz - zc))
                num_i += w * e * math.sin(k * (zz - zc))
                den += w * e
        mag = math.hypot(num_r, num_i)
        if mag <= _TOL or abs(den) <= _TOL:
            self.rep.lossy("IMPACTZ_RF_FORM_FACTOR",
                           "the synthetic RF profile integrates to zero; the amplitude "
                           "could not be calibrated", element=el.name, kind=el.kind)
            return 0.0, 0.0
        arg = math.atan2(num_i, num_r)
        sign = 1.0 if ref.species.charge >= 0 else -1.0
        amp = sign * V / mag                      # so that q·∫Ez·|F|/∫Ez = V
        theta = phase_rad - arg - 2.0 * math.pi * freq * t_center
        return (math.degrees(theta) + 180.0) % 360.0 - 180.0, amp

    def _rfdata_file(self, rows, *, derivative: bool) -> int:
        """Register an ``rfdataN.in`` body, re-using an identical one."""
        body = "\n".join(" ".join(fmt(v) for v in r) for r in rows) + "\n"
        h = hashlib.sha256(body.encode()).hexdigest()
        if h in self._rfdata_by_hash:
            return self._rfdata_by_hash[h]
        fid = self._next_rfdata
        self._next_rfdata += 1
        if fid > 999:
            self.rep.lossy("IMPACTZ_RFDATA_LIMIT",
                           "IMPACT-Z reads at most 999 rfdata files (Data.f90:178)")
        self.rfdata[fid] = body
        self._rfdata_by_hash[h] = fid
        return fid

    def _map_profile(self, el: FieldMap) -> tuple[list[float], list[float]] | None:
        """On-axis ``Ez(z)`` for a field map: our own ``native['impactz']`` table
        first, then :mod:`lattix.ir.fieldmap` when the files are readable."""
        rows = el.meta.get("impactz_rfdata")
        if rows and len(rows[0]) >= 2:
            return [r[0] for r in rows], [r[1] for r in rows]
        try:
            from lattix.ir.fieldmap import FieldMapData
        except Exception:                                   # pragma: no cover
            return None
        map_dir = el.meta.get("map_dir") or el.meta.get("field_map_dir")
        base = el.meta.get("map_base") or (el.files[0] if el.files else None)
        if not (map_dir and base and el.geom is not None):
            return None
        try:
            data = FieldMapData.load(el.geom, map_dir, base)
            prof = data.ez_profile(el.ke)
        except Exception as e:                              # noqa: BLE001
            self.rep.lossy("FM_READ_FAILED", f"could not read the field map files: {e}",
                           element=el.name, kind="FieldMap")
            return None
        if prof is None:
            return None
        z, ez = prof
        return [float(v) for v in z], [float(v) for v in ez]

    # ------------------------------------------------------------------
    # thin-gap length bookkeeping
    # ------------------------------------------------------------------
    def _absorb_gap_lengths(self, placed: list[Placed], cards: list[_Card],
                            rep: FidelityReport) -> None:
        """Take each inserted thin-gap length out of the neighbouring drift cards so
        that ``Σ length`` is preserved (PLAN invariant I-1)."""
        for idx, c in enumerate(cards):
            if c.itype != 104 or c.kind != "RFCavity":
                continue
            gap = c.length
            if not any(name == c.name for name, _ in self._pending_gaps):
                continue
            half = gap / 2.0
            before = _nearest_drift(cards, idx, -1)
            after = _nearest_drift(cards, idx, +1)
            take_b = min(half, cards[before].length) if before is not None else 0.0
            take_a = min(gap - take_b, cards[after].length) if after is not None else 0.0
            if take_b + take_a < gap - 1e-15 and before is not None:
                take_b = min(gap - take_a, cards[before].length)
            if before is not None and take_b:
                cards[before].length -= take_b
                cards[before].nseg = self._nseg(cards[before].length)
            if after is not None and take_a:
                cards[after].length -= take_a
                cards[after].nseg = self._nseg(cards[after].length)
            missing = gap - take_b - take_a
            if missing > 1e-15:
                rep.lossy("THIN_GAP_ADDS_LENGTH",
                          f"the {gap:.15g} m synthetic cavity could not be absorbed by "
                          f"the neighbouring drifts; the lattice grows by {missing:.15g} m",
                          element=c.name, kind="RFCavity", added_length_m=missing)


def _needs_errors(el: Element) -> bool:
    """True when the element needs IMPACT-Z's ``flagerr = 1`` to be modelled at all:
    a body shift, or a quadrupole roll (which shares the rotation-error column)."""
    if el.shift is not None and not el.shift.is_zero():
        return True
    mp = getattr(el, "multipole", None)
    return bool(mp is not None and any(mp.tilt.values()))


def _nearest_drift(cards: list[_Card], idx: int, step: int) -> int | None:
    j = idx + step
    while 0 <= j < len(cards):
        if cards[j].itype == 0 and cards[j].length > 0:
            return j
        if cards[j].length > 0:
            return None
        j += step
    return None


# ---------------------------------------------------------------------------
# synthetic on-axis profiles
# ---------------------------------------------------------------------------
def _raised_cosine(length: float, active: float, n: int) -> tuple[list[float], list[float]]:
    """``Ez ∝ 1 − cos(2πu)`` over the active window, 0 in the flanks.

    A profile that vanishes at both ends has no hard-edge radial impulse: the
    entrance/exit terms ``dlt`` of ``maplinear_SC`` (``SC.f90:174-193``) are
    proportional to ``Ez`` there.
    """
    n = max(9, int(n) | 1)
    z0 = 0.5 * (length - active)
    z = [length * i / (n - 1) for i in range(n)]
    ez = []
    for zz in z:
        u = (zz - z0) / active if active > 0 else 0.0
        ez.append(0.0 if u <= 0.0 or u >= 1.0 else 1.0 - math.cos(2.0 * math.pi * u))
    return z, ez


def _cell_train(length: float, n_cell: int, *, pi_mode: bool) -> tuple[list[float], list[float]]:
    """``n_cell`` half-wave cells: π mode alternates sign, 2π mode does not."""
    n_cell = max(1, int(n_cell))
    n = max(17, 16 * n_cell + 1)
    z = [length * i / (n - 1) for i in range(n)]
    ez = []
    for zz in z:
        u = zz / length
        if pi_mode:
            ez.append(math.sin(n_cell * math.pi * u))
        else:
            ez.append(1.0 - math.cos(2.0 * n_cell * math.pi * u))
    return z, ez


def _with_derivative(z, ez) -> list[tuple[float, float, float, float]]:
    """``read1_Data`` rows ``z, Ez, Ez', Ez''`` (``Data.f90:183``); ``Ez''`` is unused
    by ``getaxfldE_SC`` (it sets ``ezpp1 = 0``) but the file has four columns."""
    z = list(z)
    ez = list(ez)
    n = len(z)
    d = []
    for i in range(n):
        if i == 0:
            d.append((ez[1] - ez[0]) / (z[1] - z[0]) if n > 1 else 0.0)
        elif i == n - 1:
            d.append((ez[-1] - ez[-2]) / (z[-1] - z[-2]))
        else:
            d.append((ez[i + 1] - ez[i - 1]) / (z[i + 1] - z[i - 1]))
    return [(z[i], ez[i], d[i], 0.0) for i in range(n)]
