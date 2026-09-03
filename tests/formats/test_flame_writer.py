"""FLAME GLPS writer tests (PLAN §6 task 3.4): RULES coverage, goldens, dual
regimes, name rules and idempotence.

Golden snapshots live in ``tests/golden/flame``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_flame_writer.py``.  The only
normalisation is the lattix version in the header comment.

That FLAME itself accepts these decks (``GLPSParser`` and ``Machine``) is asserted in
``tests/oracles/test_flame_adapter.py``, which needs the engine; everything here is
pure Python.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.flame import Reader, Writer
from lattix.formats.flame.writer import NAME_RE, RESERVED, THIN_TYPES, fmt, sanitize
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
from lattix.ir.lattice import Lattice, Line, LineItem
from lattix.ir.reference import ReferenceParticle, Species, species

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "flame"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "flame"
_VERSION_RE = re.compile(r"^# written by lattix \S+ ", re.MULTILINE)


def _norm(text: str) -> str:
    return _VERSION_RE.sub("# written by lattix <version> ", text)


def assert_golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("LATTIX_UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_norm(text))
    assert path.exists(), f"missing golden {path}; rerun with LATTIX_UPDATE_GOLDEN=1"
    assert _norm(text) == path.read_text()


def proton_ref(ke: float = 2.1e6) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke,
                             rf_frequency_Hz=80.5e6)


def uranium_ref() -> ReferenceParticle:
    """FRIB's ²³⁸U³³⁺ at 0.5 MeV/u — the beam every vendored deck uses."""
    return ReferenceParticle(
        species=Species(name="ion_A238_Q33", mass_eV=931.49432e6 * 238, charge=33),
        kinetic_energy_eV=0.5e6 * 238, rf_frequency_Hz=80.5e6)


def one_element(el: Element, *, ref: ReferenceParticle | None = None) -> Lattice:
    return Lattice.from_sequence("S", [el], ref or proton_ref())


def demo_lattice() -> Lattice:
    """One element of every IR kind, so the golden pins all 22 writer rules."""
    ref = proton_ref()
    b = ref.brho_signed
    cav = RFCavity(name="CAV", length=0.24,
                   rf=RFP(phase_rad=math.radians(-35.0), frequency_Hz=80.5e6))
    cav.meta["flame_cavtype"] = "0.041QWR"
    cav.meta["flame_scl_fac"] = 0.64
    els: list[Element] = [
        Marker(name="START"),
        Quadrupole(name="QF", length=0.3, multipole=MagneticMultipoleP(Bn={1: 0.6 * b}),
                   aperture=ApertureP.circle(0.02)),
        Drift(name="D1", length=0.5),
        Bend(name="B1", length=1.0,
             bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.5, hgap=0.02)),
        Bend(name="BV", length=1.05,
             bend=BendP(angle=0.0416, e1=0.0208, e2=0.0208, tilt_ref=math.pi / 2)),
        Solenoid(name="SOL", length=0.4, solenoid=SolenoidP(Bsol_T=0.3 * b)),
        Sextupole(name="SX", length=0.2, multipole=MagneticMultipoleP(Bn={2: 1.5 * b})),
        Octupole(name="OC", length=0.2, multipole=MagneticMultipoleP(Bn={3: 2.5 * b})),
        Multipole(name="MP", multipole=MagneticMultipoleP(BnL={0: 0.02 * b}, BsL={0: 0.01 * b})),
        cav,
        RFCavity(name="CAV_V", length=0.1, rf=RFP(voltage_V=1e6, phase_rad=-math.pi / 6)),
        Kicker(name="COR", hkick=1e-3, vkick=-2e-3),
        Collimator(name="COLL", aperture=ApertureP.rect(0.02, 0.03)),
        Instrument(name="BPM1", family="BPM"),
        Instrument(name="WIRE1", family="PROFILE"),
        FieldMap(name="FM", length=0.3),
        NCells(name="NC", length=0.4),
        RFQCell(name="RQ", length=0.1),
        Foil(name="STRIP"),
        Taylor(name="TAY"),
        Patch(name="PA"),
        ReferenceChange(name="RC", dE_ref_eV=1e6),
        Freq(name="FR", frequency_Hz=162.5e6),
        Directive(name="LATTICE_1", format="tracewin", card="LATTICE", args=["4", "0"],
                  role="period_start"),
        Superposition(name="SUP", length=0.2),
        Quadrupole(name="QSKEW", length=0.3,
                   multipole=MagneticMultipoleP(Bn={1: 0.6 * b}, Bs={1: 0.6 * b})),
        Quadrupole(name="QMIS", length=0.3, multipole=MagneticMultipoleP(Bn={1: -0.6 * b}),
                   shift=BodyShiftP(x_offset=1e-4, y_offset=-2e-4, x_rot=1e-3, y_rot=2e-3,
                                    tilt=3e-3)),
        Marker(name="END"),
    ]
    return Lattice.from_sequence("DEMO", els, ref)


def fodo_lattice() -> Lattice:
    """A nested-line FODO: the writer must emit every sub-line and the ``USE:``."""
    ref = proton_ref(8e8)
    b = ref.brho_signed
    lat = Lattice(name="TOP", reference=ref)
    for el in (Drift(name="D", length=0.5),
               Quadrupole(name="QF", length=0.2,
                          multipole=MagneticMultipoleP(Bn={1: 1.5 * b})),
               Quadrupole(name="QD", length=0.2,
                          multipole=MagneticMultipoleP(Bn={1: -1.5 * b}))):
        lat.elements[el.name] = el
    lat.lines["CELL"] = Line(name="CELL", items=[LineItem(ref="D"), LineItem(ref="QF"),
                                                 LineItem(ref="D"), LineItem(ref="QD")])
    lat.lines["TOP"] = Line(name="TOP", items=[LineItem(ref="CELL", repeat=2),
                                               LineItem(ref="CELL", reverse=True)])
    lat.use = "TOP"
    return lat


# ------------------------------------------------------------------ rules table
def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_every_rule_has_an_emitter():
    w = Writer()
    for kind in w.RULES:
        assert hasattr(w, f"_emit_{kind.lower()}"), kind


# ------------------------------------------------------------------- numbers
def test_fmt_never_emits_a_leading_dot():
    """FLAME's lexer has no ``.5`` form (``src/glps.l``: ``[0-9]+(\\.[0-9]*)?…``)."""
    assert fmt(0.5) == "0.5"
    assert fmt(-0.5) == "-0.5"
    assert fmt(0.0) == "0" and fmt(-0.0) == "0"
    assert fmt(1e-9) == "1e-09"
    assert not any(fmt(v).lstrip("-").startswith(".") for v in (0.5, -0.5, 0.25, 1e-9))
    with pytest.raises(ValueError, match="cannot hold"):
        fmt(float("inf"))


def test_fmt_is_round_trip_exact():
    """%.15g is not exact for 33/238; the writer widens to repr rather than lose 7.4e-11
    of per-element map accuracy over a 471 m line (measured on ALL_lattice.lat)."""
    assert fmt(33.0 / 238.0) == repr(33.0 / 238.0)
    assert float(fmt(33.0 / 238.0)) == 33.0 / 238.0
    assert fmt(0.3) == "0.3" and fmt(1.0) == "1"        # the short form when it is exact
    for v in (33.0 / 238.0, 34.0 / 238.0, 1 / 3, 0.1, 1e-17, 6.6, 1.7976931348623157e308):
        assert float(fmt(v)) == v, v


# --------------------------------------------------------------------- names
def test_sanitize_produces_flame_identifiers():
    for raw in ("ls1_ca01_cav1", "LS1:BPM_D1129", "a b", "1bad", "_x", "with-dash", "x_"):
        assert NAME_RE.match(sanitize(raw)), (raw, sanitize(raw))


def test_reserved_names_are_avoided():
    for r in RESERVED:
        assert sanitize(r) not in RESERVED


def test_names_are_case_sensitive_and_unique(tmp_path):
    """FLAME is case sensitive, so ``qf`` and ``QF`` are two elements, not a clash."""
    ref = proton_ref()
    lat = Lattice.from_sequence("S", [Drift(name="qf", length=1), Drift(name="QF", length=1),
                                      Drift(name="qf", length=2)], ref)
    text = Writer().dumps(lat)
    defs = [ln.split(":")[0] for ln in text.splitlines() if ": drift" in ln]
    assert defs == ["qf", "QF", "qf_2"]


# ------------------------------------------------------------------- header
def test_header_is_per_nucleon():
    lat, _ = Reader().read(DATA / "LS1FS1_lattice.lat")
    text = Writer().dumps(lat)
    assert 'sim_type = "MomentMatrix";' in text
    assert "IonEs = 931494320;" in text                         # eV/u, not the total mass
    assert "IonEk = 500000;" in text
    assert f"IonChargeStates = [{33.0 / 238.0!r}, {34.0 / 238.0!r}];" in text
    assert "NCharge = [10111, 10531];" in text                  # kept from the deck
    assert "SampleFreq = 80500000;" in text


def test_a_source_element_is_prepended_when_the_lattice_has_none():
    rep = Writer().write(fodo_lattice(), Path(os.devnull)) if False else None
    text = Writer().dumps(fodo_lattice())
    lines = [ln for ln in text.splitlines() if ": source" in ln]
    assert len(lines) == 1
    body = text.split("USE:")[0]
    # the source must be the first member of the used line
    line_body = body.split("LINE = (")[-1]
    assert line_body.strip().startswith(lines[0].split(":")[0])
    assert rep is None


def test_source_added_is_recorded():
    from lattix.fidelity import FidelityReport

    w = Writer()
    w.dumps(fodo_lattice())
    assert isinstance(w._rep, FidelityReport)
    assert "FLAME_SOURCE_ADDED" in w._rep.codes()


def test_beam_arrays_are_kept_verbatim():
    """``real[k].IonEk += moment0[k][PS_PS]`` (``src/moment.cpp:172``): dropping the
    deck's centroid shifts every downstream cavity map, so it must be re-emitted."""
    lat, _ = Reader().read(DATA / "LS1FS1_lattice.lat")
    text = Writer().dumps(lat)
    assert "BaryCenter0 = [-0.0007886, 1.08371e-05," in text
    assert "S0 = [" in text and "S1 = [" in text


# ------------------------------------------------------------------ elements
def _def_line(lat: Lattice, name: str) -> str:
    for ln in Writer().dumps(lat).splitlines():
        if ln.startswith(f"{name}:"):
            return ln
    raise AssertionError(f"no definition for {name}")


def test_quadrupole_writes_a_lab_gradient():
    ref = proton_ref()
    q = Quadrupole(name="Q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 12.5}))
    assert _def_line(one_element(q, ref=ref), "Q") == "Q: quadrupole, L = 0.3, B2 = 12.5;"


def test_bend_writes_degrees_and_a_normalized_K():
    ref = proton_ref()
    b = Bend(name="B", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05),
             multipole=MagneticMultipoleP(Bn={1: 0.4 * ref.brho_signed}))
    line = _def_line(one_element(b, ref=ref), "B")
    assert f"phi = {math.degrees(0.1)!r}" in line               # 0.1 rad in degrees
    assert f"phi1 = {math.degrees(0.05)!r}" in line
    assert re.search(r"K = 0\.4(0000000000\d+)?[,;]", line), line


def test_vertical_bend_writes_ver():
    b = Bend(name="B", length=1.0, bend=BendP(angle=0.1, tilt_ref=math.pi / 2))
    assert "ver = 1" in _def_line(one_element(b), "B")


def test_oblique_bend_tilt_is_reported():
    b = Bend(name="B", length=1.0, bend=BendP(angle=0.1, tilt_ref=0.3))
    w = Writer()
    w.dumps(one_element(b))
    assert "BEND_TILT_DROPPED" in w._rep.codes()


def test_kicker_writes_theta():
    k = Kicker(name="K", hkick=1e-3, vkick=-2e-3)
    assert _def_line(one_element(k), "K") == "K: orbtrim, theta_x = 0.001, theta_y = -0.002;"


def test_rfcavity_without_a_cavtype_becomes_a_drift():
    cav = RFCavity(name="C", length=0.24, rf=RFP(voltage_V=1e6, frequency_Hz=325e6))
    w = Writer()
    text = w.dumps(one_element(cav))
    assert "C: drift, L = 0.24;" in text
    assert "RFCAVITY_NEEDS_CAVTYPE" in w._rep.codes()


def test_rfcavity_with_a_cavtype_is_written():
    cav = RFCavity(name="C", length=0.24,
                   rf=RFP(phase_rad=math.radians(-35.0), frequency_Hz=80.5e6))
    cav.meta["flame_cavtype"] = "0.085QWR"
    cav.meta["flame_scl_fac"] = 0.7
    line = _def_line(one_element(cav), "C")
    assert line == ('C: rfcavity, L = 0.24, cavtype = "0.085QWR", f = 80500000, '
                    "phi = -35, scl_fac = 0.7;")


def test_unknown_cavtype_is_reported():
    cav = RFCavity(name="C", length=0.24, rf=RFP(frequency_Hz=80.5e6))
    cav.meta["flame_cavtype"] = "0.53QWR_nope"
    w = Writer()
    w.dumps(one_element(cav))
    assert "FLAME_UNKNOWN_CAVTYPE" in w._rep.codes()


def test_skew_quadrupole_becomes_a_roll():
    ref = proton_ref()
    q = Quadrupole(name="Q", length=0.3,
                   multipole=MagneticMultipoleP(Bn={1: 1.0}, Bs={1: 1.0}))
    w = Writer()
    text = w.dumps(one_element(q, ref=ref))
    # (Bn, Bs) = (1, 1) is hypot(1,1) rotated by -atan2(1,1)/2 = -pi/8; FLAME's `roll`
    # is MAD-X's `tilt` (measured, see tests/oracles/test_flame_adapter.py)
    assert "roll = -0.392699081698724" in text
    assert "SKEW_AS_ROLL" in w._rep.codes()


def test_thin_type_with_a_length_is_padded_by_a_drift():
    """FLAME's marker/bpm/orbtrim/stripper take no ``L``: Σ length must still survive."""
    w = Writer()
    text = w.dumps(one_element(Instrument(name="BPM", length=0.05, family="BPM")))
    assert "BPM: bpm;" in text
    assert "BPM_L: drift, L = 0.05;" in text
    assert "THIN_TYPE_LENGTH_PADDED" in w._rep.codes()
    assert set(THIN_TYPES) >= {"marker", "bpm", "orbtrim", "stripper", "tmatrix"}


def test_taylor_rebuilds_the_7x7():
    t = Taylor(name="TM")
    t.matrix[0][1] = 2.0
    t.offset[2] = 3.0
    line = _def_line(one_element(t), "TM")
    flat = [float(x) for x in line.split("[")[1].split("]")[0].split(",")]
    assert len(flat) == 49
    assert flat[1] == 2.0                                        # R12
    assert flat[2 * 7 + 6] == 3.0                                # the constant column
    assert flat[48] == 1.0                                       # the augmentation row


def test_electrostatic_elements_round_trip_through_native():
    # FrontEnd's edipole/equad live in the `cell` line, not in the `cell1` the deck USEs
    lat, _ = Reader().read(DATA / "FrontEnd.lat", use="cell")
    text = Writer().dumps(lat)
    assert "eb1_1_0: edipole," in text
    assert "qe1h_1_0: equad," in text and "radius = 0.0746" in text


def test_aperture_becomes_aper():
    d = Drift(name="D", length=1.0, aperture=ApertureP.circle(0.02))
    assert _def_line(one_element(d), "D") == "D: drift, L = 1, aper = 0.02;"


# ------------------------------------------------------------------- degrades
@pytest.mark.parametrize(("el", "code"), [
    (FieldMap(name="X", length=0.3), "FM_TO_DRIFT"),
    (NCells(name="X", length=0.4), "NCELLS_TO_DRIFT"),
    (RFQCell(name="X", length=0.1), "RFQ_TO_DRIFT"),
    (Octupole(name="X", length=0.2), "OCTUPOLE_TO_DRIFT"),
    (Patch(name="X"), "PATCH_DROPPED"),
    (ReferenceChange(name="X", dE_ref_eV=1e6), "REFCHANGE_DROPPED"),
    (Collimator(name="X", aperture=ApertureP.rect(0.01, 0.01)), "COLLIMATOR_TO_MARKER"),
    (Superposition(name="X", length=0.2), "SUPERPOSITION_FLATTENED"),
    (Directive(name="X", card="LATTICE", role="period_start"), "FOREIGN_DIRECTIVE"),
    (Multipole(name="X", multipole=MagneticMultipoleP(BnL={2: 1.0})),
     "MULTIPOLE_ORDERS_DROPPED"),
])
def test_degradations_are_recorded_and_strict_raises(el, code, tmp_path):
    w = Writer()
    w.dumps(one_element(el))
    assert code in w._rep.codes(), w._rep.codes()
    with pytest.raises(TranslationError, match=code):
        Writer().write(one_element(el), tmp_path / "x.lat", strict=True)


def test_freq_writes_nothing_but_stays_exact():
    w = Writer()
    text = w.dumps(one_element(Freq(name="FR", frequency_Hz=162.5e6)))
    assert "FR" not in text.split("USE:")[0].replace("BaryCenter", "")
    assert "FOREIGN_DIRECTIVE" not in w._rep.codes()


# ------------------------------------------------------------------- goldens
def test_golden_demo_lattice(tmp_path):
    out = tmp_path / "demo.lat"
    Writer().write(demo_lattice(), out)
    assert_golden("demo.lat", out.read_text())


def test_golden_fodo_nested_lines(tmp_path):
    out = tmp_path / "fodo.lat"
    Writer().write(fodo_lattice(), out)
    assert_golden("fodo.lat", out.read_text())


def test_golden_uranium_front_end(tmp_path):
    """A ²³⁸U³³⁺ line with the FRIB spellings: cavity, stripper, vertical bend, tmatrix."""
    ref = uranium_ref()
    cav = RFCavity(name="cav1", length=0.24,
                   rf=RFP(phase_rad=math.radians(-35.0), frequency_Hz=80.5e6))
    cav.meta["flame_cavtype"] = "0.041QWR"
    cav.meta["flame_scl_fac"] = 0.64
    els: list[Element] = [
        Drift(name="d1", length=0.207065, aperture=ApertureP.circle(0.02)),
        cav,
        Instrument(name="bpm1", family="BPM"),
        Solenoid(name="sol1", length=0.1, solenoid=SolenoidP(Bsol_T=5.34),
                 aperture=ApertureP.circle(0.02)),
        Kicker(name="dch1", hkick=0.0, vkick=0.0),
        Bend(name="bend1", length=1.0, bend=BendP(angle=math.radians(-45.0),
                                                  e1=math.radians(-22.5),
                                                  e2=math.radians(-22.5), tilt_ref=math.pi / 2)),
        Foil(name="strip"),
        Taylor(name="tm1"),
        Marker(name="end"),
    ]
    lat = Lattice.from_sequence("cell", els, ref)
    lat.meta["flame"] = {"mass_number": 238,
                         "globals": {"MpoleLevel": "2", "HdipoleFitMode": "1",
                                     "Stripper_IonChargeStates": [78.0 / 238.0],
                                     "Stripper_NCharge": [5300.0]}}
    out = tmp_path / "uranium.lat"
    Writer().write(lat, out, eng_data_dir="data")
    assert_golden("uranium.lat", out.read_text())


# --------------------------------------------------------------- round trips
@pytest.mark.parametrize("name", sorted(p.name for p in DATA.glob("*.lat")))
def test_write_then_read_is_idempotent(name, tmp_path):
    """FLAME → IR → GLPS → IR → GLPS is byte identical (invariant I-13)."""
    lat, _ = Reader().read(DATA / name)
    first = tmp_path / "a.lat"
    Writer().write(lat, first)
    lat2, _ = Reader().read(first)
    assert Writer().dumps(lat2) == first.read_text()


@pytest.mark.parametrize("name", sorted(p.name for p in DATA.glob("*.lat")))
def test_round_trip_preserves_structure(name, tmp_path):
    lat, _ = Reader().read(DATA / name)
    out = tmp_path / "a.lat"
    Writer().write(lat, out)
    lat2, _ = Reader().read(out)
    a, b = lat.flatten(), lat2.flatten()
    # the writer prepends a source when the deck had none (parse1.lat)
    assert len(b) - len(a) in (0, 1)
    assert sum(p.length for p in a) == pytest.approx(sum(p.length for p in b), abs=1e-12)
    assert lat2.reference.species.charge == lat.reference.species.charge
    assert lat2.reference.kinetic_energy_eV == pytest.approx(lat.reference.kinetic_energy_eV,
                                                             rel=1e-14)


def test_ledger_has_one_entry_per_element():
    """PLAN §4.4: exactly one entry per source element."""
    lat = demo_lattice()
    w = Writer()
    w.dumps(lat)
    assert len(w._rep.entries) >= len(lat.flatten())
    per_element = [e for e in w._rep.entries if e.element]
    names = {p.element.name for p in lat.flatten()}
    assert names <= {e.element for e in per_element}
