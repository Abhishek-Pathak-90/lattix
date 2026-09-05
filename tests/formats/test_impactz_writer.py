"""IMPACT-Z writer tests (PLAN §6 task 3.3): goldens, rules coverage, dual regimes,
column-by-column checks of the type-code mapping.

Golden snapshots live in ``tests/golden/impactz``; regenerate them deliberately with
``LATTIX_UPDATE_GOLDEN=1 pytest tests/formats/test_impactz_writer.py``.  The only
normalisation applied is the lattix version in the header comment.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.impactz import Reader, Writer
from lattix.formats.impactz.reader import parse_deck
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
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
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.testing import needs

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "impactz"
_VERSION_RE = re.compile(r"^! IMPACT-Z deck written by lattix \S+ from ", re.MULTILINE)


def _norm(text: str) -> str:
    return _VERSION_RE.sub("! IMPACT-Z deck written by lattix <version> from ", text)


def assert_golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("LATTIX_UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_norm(text))
    assert path.exists(), f"missing golden {path}; rerun with LATTIX_UPDATE_GOLDEN=1"
    assert _norm(text) == path.read_text()


def proton_ref(ke: float = 2.1e6, freq: float = 162.5e6) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke,
                             rf_frequency_Hz=freq)


def demo_linac() -> Lattice:
    """Drifts, quads, a solenoid, a bend, a thin cavity, a kicker, a slit, a marker."""
    ref = proton_ref()
    els = [
        Drift(name="D1", length=0.3, aperture=ApertureP.circle(0.014)),
        Quadrupole(name="QF", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0})),
        Drift(name="D2", length=0.4),
        Solenoid(name="SOL", length=0.4, solenoid=SolenoidP(Bsol_T=0.5)),
        Drift(name="D3", length=0.4),
        RFCavity(name="BUNCH", length=0.0,
                 rf=RFP(voltage_V=1e6, phase_rad=-math.pi / 6, frequency_Hz=162.5e6)),
        Drift(name="D4", length=0.4),
        Quadrupole(name="QD", length=0.2, multipole=MagneticMultipoleP(Bn={1: -5.0})),
        Drift(name="D5", length=0.3),
        Bend(name="B1", length=1.0,
             bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.45, hgap=0.03)),
        Kicker(name="COR", hkick=1e-3, vkick=-2e-3),
        Collimator(name="SLIT", aperture=ApertureP.rect(0.01, 0.02)),
        Marker(name="END"),
    ]
    return Lattice.from_sequence("demo", els, ref)


def one(el, *, ref: ReferenceParticle | None = None) -> Lattice:
    return Lattice.from_sequence("s", [el], ref or proton_ref())


def cards_of(path: Path):
    return parse_deck(path.read_text())[1]


def write(lat: Lattice, tmp_path: Path, **kw):
    out = tmp_path / "ImpactZ.in"
    rep = Writer().write(lat, out, **kw)
    return out, rep


# ------------------------------------------------------------------ rules table
def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_every_rule_has_a_builder():
    w = Writer()
    for kind in w.RULES:
        if kind in ("Freq", "Directive"):
            continue
        assert hasattr(w, f"_w_{kind.lower()}"), kind


# ---------------------------------------------------------------------- header
def test_header_records(tmp_path):
    out, _ = write(demo_linac(), tmp_path, n_particles=250, grid=(32, 32, 64),
                   current_A=0.0)
    h, cards, _ = parse_deck(out.read_text())
    assert (h.npcol, h.nprow) == (1, 1)
    assert (h.dim, h.np, h.flagmap, h.flagerr, h.flagdiag) == (6, 250, 1, 0, 1)
    assert (h.nx, h.ny, h.nz) == (32, 32, 64)
    assert h.nchrg == 1 and h.nptlist == [250]
    assert h.kinetic_energy_eV == pytest.approx(2.1e6)
    assert h.mass_eV == pytest.approx(species("proton").mass_eV)
    assert h.charge == pytest.approx(1.0)
    assert h.frequency_Hz == pytest.approx(162.5e6)
    assert h.current_A == 0.0 and h.currlist == [0.0]
    assert h.qmcclist[0] == pytest.approx(1.0 / species("proton").mass_eV)
    assert cards[-1].itype == -99


def test_h_minus_charge_is_minus_one(tmp_path):
    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=2.1e6,
                            rf_frequency_Hz=162.5e6)
    out, _ = write(one(Drift(name="d", length=1.0), ref=ref), tmp_path)
    h, _, _ = parse_deck(out.read_text())
    assert h.charge == pytest.approx(-1.0)
    assert h.qmcclist[0] == pytest.approx(-1.0 / species("h-").mass_eV)


def test_frequency_taken_from_the_first_rf_element(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
    lat = Lattice.from_sequence("s", [
        Drift(name="d", length=1.0),
        RFCavity(name="c", rf=RFP(voltage_V=1e6, frequency_Hz=352.21e6)),
    ], ref)
    out, _ = write(lat, tmp_path)
    assert parse_deck(out.read_text())[0].frequency_Hz == pytest.approx(352.21e6)


def test_no_frequency_anywhere_is_lossy(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
    out, rep = write(one(Drift(name="d", length=1.0), ref=ref), tmp_path)
    assert "IMPACTZ_NO_FREQUENCY" in rep.codes()
    assert parse_deck(out.read_text())[0].frequency_Hz == pytest.approx(1e9)


# ------------------------------------------------------------- element columns
def test_drift_columns(tmp_path):
    out, _ = write(one(Drift(name="d", length=1.5, aperture=ApertureP.circle(0.02))),
                   tmp_path, steps_per_m=4.0, map_steps=7)
    c = cards_of(out)[0]
    assert (c.length, c.nseg, c.mapstp, c.itype) == (1.5, 6, 7, 0)
    assert c.v(1) == pytest.approx(0.02)


def test_quadrupole_writes_the_lab_gradient(tmp_path):
    q = Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: -7.25}))
    c = cards_of(write(one(q), tmp_path)[0])[0]
    assert c.itype == 1
    assert c.v(1) == pytest.approx(-7.25)     # T/m, lab field
    assert c.v(2) == 0.0                      # file ID: 0 = analytic hard edge
    assert c.v(3) == pytest.approx(0.014)     # default radius


def test_skew_quadrupole_is_lossy_and_uses_the_rotation_column(tmp_path):
    q = Quadrupole(name="q", length=0.3,
                   multipole=MagneticMultipoleP(Bn={1: 5.0}, tilt={1: math.pi / 4}))
    out, rep = write(one(q), tmp_path)
    assert "IMPACTZ_SKEW_QUAD" in rep.codes()
    assert cards_of(out)[0].v(8) == pytest.approx(math.pi / 4)
    # the rotation column only bites when flagerr = 1, so a rolled quad turns it on
    assert parse_deck(out.read_text())[0].flagerr == 1


def test_solenoid_writes_the_lab_field(tmp_path):
    s = Solenoid(name="s", length=0.4, solenoid=SolenoidP(Bsol_T=0.5))
    c = cards_of(write(one(s), tmp_path)[0])[0]
    assert c.itype == 3 and c.v(1) == pytest.approx(0.5)


def test_bend_columns_are_radians_halfgap_and_fint(tmp_path):
    b = Bend(name="b", length=1.0,
             bend=BendP(angle=0.1, e1=0.05, e2=0.06, edge_int1=0.45, hgap=0.03))
    b.multipole.Bn[1] = 0.0
    c = cards_of(write(one(b), tmp_path)[0])[0]
    assert c.itype == 4
    assert c.v(1) == pytest.approx(0.1)      # angle [rad] — NOT degrees
    assert c.v(2) == 0.0                     # k1
    assert c.v(3) > 100                      # file ID > 100 selects the z map
    assert c.v(4) == pytest.approx(0.03)     # HALF gap (AccSimulator hgap = 2·Param(5))
    assert c.v(5) == pytest.approx(0.05)     # e1 [rad]
    assert c.v(6) == pytest.approx(0.06)     # e2 [rad]
    assert c.v(9) == pytest.approx(0.45)     # fint


def test_bend_k1_is_normalised_with_the_entrance_rigidity(tmp_path):
    ref = proton_ref()
    b = Bend(name="b", length=1.0, bend=BendP(angle=0.1))
    b.multipole.Bn[1] = 2.0
    c = cards_of(write(one(b, ref=ref), tmp_path)[0])[0]
    assert c.v(2) == pytest.approx(2.0 / ref.brho_signed)


def test_vertical_bend_is_lossy(tmp_path):
    b = Bend(name="b", length=1.0, bend=BendP(angle=0.1, tilt_ref=math.pi / 2))
    _, rep = write(one(b), tmp_path)
    assert "IMPACTZ_NO_REF_TILT" in rep.codes()


def test_sextupole_and_octupole_use_type_5(tmp_path):
    sx = Sextupole(name="sx", length=0.2, multipole=MagneticMultipoleP(Bn={2: 120.0}))
    oc = Octupole(name="oc", length=0.2, multipole=MagneticMultipoleP(Bn={3: 90.0}))
    lat = Lattice.from_sequence("s", [sx, oc], proton_ref())
    out, rep = write(lat, tmp_path)
    a, b = cards_of(out)[:2]
    assert (a.itype, a.v(1), a.v(2)) == (5, 2.0, 120.0)
    assert (b.itype, b.v(1), b.v(2)) == (5, 3.0, 90.0)
    # integrator="auto" (the default) switches to the Lorentz integrator (flagmap = 2)
    # as soon as a type-5 is written: the linear map returns NaN for it (measured)
    assert "IMPACTZ_MULTIPOLE_LINEAR_MAP" not in rep.codes()
    assert parse_deck(out.read_text())[0].flagmap == 2
    # the linear-map integrator reads Param(2) — the order id — as the gradient
    out, rep = write(lat, tmp_path, integrator="map")
    assert rep.codes()["IMPACTZ_MULTIPOLE_LINEAR_MAP"] == 2
    assert parse_deck(out.read_text())[0].flagmap == 1


def test_thin_multipole_uses_the_minus_55_kick(tmp_path):
    ref = proton_ref()
    m = Multipole(name="m", multipole=MagneticMultipoleP(BnL={1: 0.5, 2: 3.0}))
    c = cards_of(write(one(m, ref=ref), tmp_path)[0])[0]
    assert c.itype == -55
    assert c.v(3) == pytest.approx(0.5 / ref.brho_signed)     # k1 = B1L/Bρ
    assert c.v(4) == pytest.approx(3.0 / ref.brho_signed)     # k2 = B2L/Bρ


def test_kicker_uses_the_minus_21_centroid_shift(tmp_path):
    c = cards_of(write(one(Kicker(name="k", hkick=1e-3, vkick=-2e-3)), tmp_path)[0])[0]
    assert c.itype == -21
    assert c.v(3) == pytest.approx(1e-3)      # dpx [rad] -> drange(4)
    assert c.v(5) == pytest.approx(-2e-3)     # dpy [rad] -> drange(6)


def test_collimator_uses_the_minus_13_slit(tmp_path):
    col = Collimator(name="c", aperture=ApertureP.rect(0.01, 0.02))
    c = cards_of(write(one(col), tmp_path)[0])[0]
    assert c.itype == -13
    assert (c.v(2), c.v(3)) == pytest.approx((-0.01, 0.01))
    assert (c.v(4), c.v(5)) == pytest.approx((-0.02, 0.02))


def test_marker_is_a_zero_length_drift(tmp_path):
    c = cards_of(write(one(Marker(name="m")), tmp_path)[0])[0]
    assert (c.itype, c.length) == (0, 0.0) and c.nseg >= 1


def test_misalignment_columns(tmp_path):
    q = Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0}),
                   shift=BodyShiftP(x_offset=1e-3, y_offset=-2e-3, tilt=0.01))
    out, _ = write(one(q), tmp_path)
    c = cards_of(out)[0]
    assert (c.v(4), c.v(5), c.v(8)) == pytest.approx((1e-3, -2e-3, 0.01))
    assert parse_deck(out.read_text())[0].flagerr == 1


def test_misalignments_are_dropped_when_flagerr_is_off(tmp_path):
    q = Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 1.0}),
                   shift=BodyShiftP(x_offset=1e-3))
    out, rep = write(one(q), tmp_path, errors=False)
    assert "IMPACTZ_MISALIGNMENT_DROPPED" in rep.codes()
    assert cards_of(out)[0].v(4) == 0.0
    assert parse_deck(out.read_text())[0].flagerr == 0


def test_negative_solenoid_dx_would_trip_the_ideal_cavity_sentinel(tmp_path):
    """MEASURED: a solenoid whose Param(5) (= dx) is negative is treated by
    ``drift1_BeamBunch`` as an ideal RF cavity — the map becomes a drift and the
    reference energy moves.  The writer must never emit that."""
    s = Solenoid(name="s", length=0.4, solenoid=SolenoidP(Bsol_T=0.5),
                 shift=BodyShiftP(x_offset=-1e-3))
    out, rep = write(one(s), tmp_path)
    assert "IMPACTZ_SOLENOID_DX_SENTINEL" in rep.codes()
    assert cards_of(out)[0].v(4) == 0.0


# ------------------------------------------------------------------- RF cavity
def test_thin_cavity_uses_the_ideal_model(tmp_path):
    lat = demo_linac()
    out, rep = write(lat, tmp_path)
    cav = next(c for c in cards_of(out) if c.itype == 104)
    assert cav.length == pytest.approx(1e-3)
    assert cav.v(1) == pytest.approx(1e6 / 1e-3)     # gradient [V/m]
    assert cav.v(2) == pytest.approx(162.5e6)        # frequency [Hz]
    assert cav.v(3) == pytest.approx(-30.0)          # SYNCHRONOUS phase [deg]
    assert cav.v(4) < 0                              # negative file ID = ideal cavity
    assert "THIN_GAP_AS_SHORT_CAVITY" in rep.codes()


def test_thin_gap_length_comes_out_of_the_neighbouring_drifts(tmp_path):
    lat = demo_linac()
    before = lat.total_length
    out, rep = write(lat, tmp_path, thin_gap_length_m=2e-3)
    cards = [c for c in cards_of(out) if c.itype != -99]
    assert sum(c.length for c in cards) == pytest.approx(before, abs=1e-12)
    assert "THIN_GAP_ADDS_LENGTH" not in rep.codes()


def test_thin_gap_without_drift_neighbours_adds_length(tmp_path):
    lat = one(RFCavity(name="c", rf=RFP(voltage_V=1e6, frequency_Hz=162.5e6)))
    _, rep = write(lat, tmp_path)
    assert "THIN_GAP_ADDS_LENGTH" in rep.codes()


def test_cavity_gradient_from_dE_ref(tmp_path):
    cav = RFCavity(name="c", length=0.5,
                   rf=RFP(frequency_Hz=1e9, phase_rad=0.0, dE_ref_eV=5e6))
    c = cards_of(write(one(cav), tmp_path)[0])[0]
    assert c.v(1) * c.length == pytest.approx(5e6)


def test_rfdata_model_writes_a_file(tmp_path):
    lat = demo_linac()
    out, rep = write(lat, tmp_path, rf_model="rfdata")
    assert (tmp_path / "rfdata1.in").is_file()
    rows = [ln.split() for ln in (tmp_path / "rfdata1.in").read_text().splitlines()]
    assert all(len(r) == 4 for r in rows)          # z, Ez, Ez', Ez''
    assert float(rows[0][1]) == pytest.approx(0.0)  # profile vanishes at both ends
    assert float(rows[-1][1]) == pytest.approx(0.0)
    cav = next(c for c in cards_of(out) if c.itype == 104)
    assert cav.v(4) == pytest.approx(1.0)          # positive file ID
    assert "IMPACTZ_RF_ABSOLUTE_PHASE" in rep.codes()


def test_rfdata_amplitude_reproduces_the_voltage(tmp_path):
    """The generated profile is calibrated so that the *transit-time corrected*
    integral ``|∫Ez e^{ik(z−z_c)}dz|`` equals V — that is what IMPACT-Z integrates."""
    ref = proton_ref()
    lat = one(RFCavity(name="c", length=0.01,
                       rf=RFP(voltage_V=2e6, phase_rad=0.0, frequency_Hz=162.5e6)),
              ref=ref)
    write(lat, tmp_path, rf_model="rfdata")
    rows = [[float(v) for v in ln.split()]
            for ln in (tmp_path / "rfdata1.in").read_text().splitlines()]
    z = [r[0] for r in rows]
    ez = [r[1] for r in rows]
    k = 2 * math.pi * 162.5e6 / (ref.beta * 299_792_458.0)
    zc = 0.5 * (z[0] + z[-1])

    def trap(f):
        return sum(0.5 * (f(i) + f(i + 1)) * (z[i + 1] - z[i]) for i in range(len(z) - 1))

    re_ = trap(lambda i: ez[i] * math.cos(k * (z[i] - zc)))
    im_ = trap(lambda i: ez[i] * math.sin(k * (z[i] - zc)))
    assert math.hypot(re_, im_) == pytest.approx(2e6, rel=1e-9)
    assert trap(lambda i: ez[i]) > 2e6      # ∫Ez is V/|form factor| > V


def test_field_map_with_real_data_becomes_an_rfdata_cavity(tmp_path):
    """A FieldMap whose TraceWin component files are readable is written as a type-104
    cavity with its own generated ``rfdataN.in`` (four columns: z, Ez, Ez', Ez'')."""
    maps = tmp_path / "maps"
    maps.mkdir()
    n = 40
    lines = [f"{n} 0.3", "1"]
    lines += [f"{math.sin(math.pi * (0.3 * i / n) / 0.3):.9g}" for i in range(n + 1)]
    (maps / "cav.edz").write_text("\n".join(lines) + "\n")
    fm = FieldMap(name="fm", length=0.3, geom=100, files=["cav"], ke=1.0,
                  rf=RFP(frequency_Hz=352.21e6, phase_rad=-0.3),
                  meta={"field_map_dir": str(maps)})
    out, rep = write(one(fm), tmp_path)
    assert "FM_AS_RFDATA" in rep.codes(), rep.summary()
    card = cards_of(out)[0]
    assert card.itype == 104 and card.v(4) == pytest.approx(1.0)
    rows = [ln.split() for ln in (tmp_path / "rfdata1.in").read_text().splitlines()]
    assert len(rows) == n + 1 and all(len(r) == 4 for r in rows)
    assert float(rows[0][0]) == 0.0 and float(rows[-1][0]) == pytest.approx(0.3)
    assert max(abs(float(r[1])) for r in rows) > 0


def test_field_map_without_data_becomes_a_drift(tmp_path):
    fm = FieldMap(name="fm", length=0.3, geom=100, files=["missing"],
                  rf=RFP(frequency_Hz=352.21e6))
    out, rep = write(one(fm), tmp_path)
    assert "FM_TO_DRIFT" in rep.codes()
    assert cards_of(out)[0].itype == 0


# ------------------------------------------------------------------ degradations
@pytest.mark.parametrize(("el", "code"), [
    (RFQCell(name="r", length=0.1), "RFQ_TO_DRIFT"),
    (Foil(name="f", length=0.0), "FOIL_DROPPED"),
    (Taylor(name="t", length=0.0), "TAYLOR_DROPPED"),
    (Patch(name="p", length=0.0), "PATCH_DROPPED"),
    (ReferenceChange(name="rc", dE_ref_eV=1e5), "REFCHANGE_DROPPED"),
    (Superposition(name="sp", length=0.5), "SUPERPOSITION_TO_DRIFT"),
    (Instrument(name="i", length=0.0), "INSTRUMENT_TO_MARKER"),
    (FieldMap(name="fm", length=0.3), "FM_TO_DRIFT"),
    (Directive(name="dv", card="SET_SYNC_PHASE", role="sync_phase"), "FOREIGN_DIRECTIVE"),
])
def test_degradations_are_recorded_in_both_regimes(el, code, tmp_path):
    out, rep = write(one(el), tmp_path)
    assert code in rep.codes(), rep.summary()
    with pytest.raises(TranslationError):
        Writer().write(one(el), tmp_path / "strict.in", strict=True)


def test_freq_card_is_exact(tmp_path):
    lat = Lattice.from_sequence("s", [
        Freq(name="f", frequency_Hz=325e6),
        RFCavity(name="c", length=0.1, rf=RFP(voltage_V=1e6, frequency_Hz=325e6)),
    ], proton_ref())
    out, rep = write(lat, tmp_path)
    assert "FREQ" not in str(rep.codes())
    assert [c.itype for c in cards_of(out)] == [104, -99]


def test_ncells_is_an_ideal_cavity_with_a_lossy_note(tmp_path):
    nc = NCells(name="n", length=0.5, params={"mode": 1, "n_cells": 4, "beta_g": 0.2},
                rf=RFP(voltage_V=2e6, phase_rad=-0.4, frequency_Hz=325e6))
    out, rep = write(one(nc), tmp_path)
    c = cards_of(out)[0]
    assert c.itype == 103 and c.v(4) < 0
    assert "NCELLS_AS_CCL" in rep.codes() and "IMPACTZ_NCELLS_PARAMS" in rep.codes()


def test_strict_passes_on_an_exactly_representable_lattice(tmp_path):
    lat = Lattice.from_sequence("s", [
        Drift(name="d", length=1.0),
        Quadrupole(name="q", length=0.3, multipole=MagneticMultipoleP(Bn={1: 3.0})),
        Solenoid(name="s", length=0.2, solenoid=SolenoidP(Bsol_T=0.1)),
        Bend(name="b", length=1.0, bend=BendP(angle=0.05)),
        Kicker(name="k", hkick=1e-4),
        Marker(name="m"),
    ], proton_ref())
    rep = Writer().write(lat, tmp_path / "ImpactZ.in", strict=True)
    assert rep.ok and rep.counts.get("LOSSY", 0) == 0


# ----------------------------------------------------------------- name tagging
def test_names_survive_the_comment_tag(tmp_path):
    lat = demo_linac()
    out, _ = write(lat, tmp_path)
    back, _ = Reader().read(out)
    assert [p.element.name for p in back.flatten()] == [p.element.name for p in lat.flatten()]


def test_odd_names_are_percent_encoded(tmp_path):
    lat = one(Drift(name="a b/c!d", length=1.0))
    out, _ = write(lat, tmp_path)
    assert "name=a%20b%2Fc%21d" in out.read_text()
    back, _ = Reader().read(out)
    assert back.flatten()[0].element.name == "a b/c!d"


# --------------------------------------------------------------------- goldens
def test_golden_demo_linac(tmp_path):
    out, _ = write(demo_linac(), tmp_path)
    assert_golden("demo_linac.ImpactZ.in", out.read_text())


@needs("madx")
def test_golden_fodo_from_madx(tmp_path):
    from lattix.formats.madx import Reader as MadxReader

    lat, _ = MadxReader().read(DATA / "helix" / "fodo.madx")
    lat.reference = lat.reference.model_copy(update={"rf_frequency_Hz": 352.21e6})
    out, rep = write(lat, tmp_path)
    assert_golden("fodo.ImpactZ.in", out.read_text())
    assert rep.ok, rep.summary()
    cards = [c for c in cards_of(out) if c.itype != -99]
    assert sum(c.length for c in cards) == pytest.approx(6.6)
    assert [c.itype for c in cards].count(4) == 2      # two sbends
    assert [c.itype for c in cards].count(1) == 2      # two quadrupoles


# ----------------------------------------------------------------- idempotence
@pytest.mark.parametrize("deck", ["Example1", "Example2", "Example3"])
def test_write_read_write_is_a_fixed_point(deck, tmp_path):
    src = DATA / "impactz" / deck / "ImpactZ.in"
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    lat, _ = Reader().read(src)
    Writer().write(lat, a / "ImpactZ.in")
    lat2, _ = Reader().read(a / "ImpactZ.in")
    Writer().write(lat2, b / "ImpactZ.in")
    assert (a / "ImpactZ.in").read_text() == (b / "ImpactZ.in").read_text()


def test_golden_is_a_fixed_point(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    Writer().write(demo_linac(), a / "ImpactZ.in")
    lat, _ = Reader().read(a / "ImpactZ.in")
    Writer().write(lat, b / "ImpactZ.in")
    lat2, _ = Reader().read(b / "ImpactZ.in")
    c = b / "c"
    c.mkdir()
    Writer().write(lat2, c / "ImpactZ.in")
    assert (b / "ImpactZ.in").read_text() == (c / "ImpactZ.in").read_text()


# -------------------------------------------------------------------- options
def test_invalid_options_raise():
    with pytest.raises(ValueError):
        Writer().write(demo_linac(), Path("/dev/null"), integrator="nope")
    with pytest.raises(ValueError):
        Writer().write(demo_linac(), Path("/dev/null"), rf_model="nope")
    with pytest.raises(ValueError):
        Writer().write(demo_linac(), Path("/dev/null"), map_steps=0)


def test_lorentz_integrator_sets_flagmap_2(tmp_path):
    out, rep = write(demo_linac(), tmp_path, integrator="lorentz")
    assert parse_deck(out.read_text())[0].flagmap == 2
    assert "IMPACTZ_IDEAL_CAVITY_NEEDS_MAP" in rep.codes()


def test_numbers_use_15_significant_digits(tmp_path):
    q = Quadrupole(name="q", length=0.3,
                   multipole=MagneticMultipoleP(Bn={1: 1.2345678901234567}))
    out, _ = write(one(q), tmp_path)
    assert "1.23456789012346" in out.read_text()
