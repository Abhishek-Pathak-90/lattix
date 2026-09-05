"""IMPACT-T writer: rules coverage, the deck, the generated field files, the measured conventions,
goldens."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import pytest

from lattix import read, write
from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.impactt import Writer
from lattix.formats.impactt.reader import parse_deck, solenoid_table_integrals
from lattix.formats.impactt.rfprofile import (
    calibrate,
    evaluate_fourier,
    fourier_coefficients,
    gain_from_profile,
    integrate_reference,
    raised_cosine,
)
from lattix.formats.impactt.writer import _SOL_EDGE_A
from lattix.ir.elements import RFP, Drift, MagneticMultipoleP, Quadrupole, RFCavity, Sextupole
from lattix.ir.lattice import Lattice
from lattix.ir.walk import propagate
from tests.formats.test_pyorbit_writer import all_kinds_lattice, ref_at

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "impactt"


def _deck(lat: Lattice, tmp_path: Path, **opts):
    out = tmp_path / "ImpactT.in"
    rep = write(lat, out, "impactt", **opts)
    header, cards, comments = parse_deck(out.read_text())
    return out, rep, header, cards


def _by_name(cards):
    out = {}
    for c in cards:
        out.setdefault(c.tag.get("name"), []).append(c)
    return out


def _normalise(text: str) -> str:
    return re.sub(r"written by lattix \S+", "written by lattix <ver>", text)


def _golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = _normalise(path.read_text())
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(), f"golden {name} differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_deck_structure_and_ledger(tmp_path):
    lat = all_kinds_lattice()
    out, rep, header, cards = _deck(lat, tmp_path)
    assert header.kinetic_energy_eV == 2.1e6 and header.mass_eV == pytest.approx(938272088.16)
    assert header.charge == 1 and header.frequency_Hz == 162.5e6 and header.dt_s == 1e-12
    by = _by_name(cards)
    assert [c.itype for c in by["q1"]] == [1] and by["q1"][0].zedge == 0.5 and by["q1"][0].v(2) == 1.2
    assert by["s1"][0].itype == 5 and by["o1"][0].itype == 5 and by["b1"][0].itype == 4
    assert by["sol1"][0].itype == 3 and [c.itype for c in by["c1"]] == [104] and by["c2"][0].itype == 104
    assert [c.itype for c in by["k1"]] == [0, -1] and [c.itype for c in by["col1"]] == [0, -11]
    assert by["k1"][1].zedge == pytest.approx(2.451) and by["col1"][1].zedge == pytest.approx(2.526)   # centres
    assert cards[-1].itype == -99 and cards[-1].zedge == pytest.approx(2.851)   # the thin gap added 1 mm
    assert by["mk1"][0].length == 0 and by["bpm1"][0].itype == 0 and by["f1"][0].itype == 0
    codes = rep.codes()
    for code in ("IMPACTT_MULTIPOLE_AS_KICK", "IMPACTT_SOLENOID_TABLE", "IMPACTT_RF_DRIVEN_PHASE",
                 "THIN_GAP_AS_SHORT_CAVITY", "THIN_GAP_ADDS_LENGTH", "THICK_KICKER_SPLIT",
                 "THICK_COLLIMATOR_AT_CENTRE", "INSTRUMENT_AS_MARKER"):
        assert code in codes, code
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["1T2.T7", "ImpactT.in", "rfdata1", "rfdata3", "rfdata4"]     # bend, solenoid, two cavities
    assert "! lattix: name=sol1 kind=Solenoid L=0.3 Bsol=0.5 pad=0.0003" in out.read_text()


def test_measured_conventions(tmp_path):
    lat = all_kinds_lattice()
    _, _, _, cards = _deck(lat, tmp_path)
    by = _by_name(cards)
    placed = {p.element.name: p for p in propagate(lat)}
    # steering: Param 3..8 = dx, dpx, dy, dpy, dz, dpz with the momenta in γβ units (the kicker
    # angle times the local γβ); the thin multipole's dipole terms kick x′ = −BnL/Bρ, y′ = +BsL/Bρ
    gb = placed["k1"].ref_in.gamma * placed["k1"].ref_in.beta
    k1 = by["k1"][1]
    assert k1.v(2) == pytest.approx(k1.zedge)
    assert k1.v(4) == pytest.approx(1e-3 * gb) and k1.v(6) == pytest.approx(-2e-3 * gb)
    gb0 = placed["m1"].ref_in.gamma * placed["m1"].ref_in.beta
    brho = placed["m1"].ref_in.brho_signed
    m1 = by["m1"][0]
    assert m1.itype == -1
    assert m1.v(4) == pytest.approx(-0.01 / brho * gb0) and m1.v(6) == pytest.approx(0.02 / brho * gb0)
    # dipole: By = Bρ_signed·θ/L_arc, the pole faces in rfdata1 (22 values, γ of the entrance)
    b1 = by["b1"][0]
    assert b1.v(3) == pytest.approx(brho * 0.1 / 1.0) and b1.v(2) == 0.0
    rows = (tmp_path / "rfdata1").read_text().split()
    assert len(rows) == 22 and float(rows[1]) == pytest.approx(placed["b1"].ref_in.gamma)
    assert float(rows[2]) == pytest.approx(math.tan(0.05))                      # k1: entrance face at e1
    # collimator: xmin xmax ymin ymax, rectangular flag ≤ 10
    col = by["col1"][1]
    assert [col.v(i) for i in (3, 4, 5, 6, 7)] == [-0.01, 0.01, -0.02, 0.02, 1]


def test_solenoid_table_is_the_second_order_edge_design(tmp_path):
    lat = all_kinds_lattice()
    _, _, _, cards = _deck(lat, tmp_path)
    sol = _by_name(cards)["sol1"][0]
    h = 0.3 / 1500
    assert sol.length == pytest.approx(0.3 + 3 * h) and sol.zedge == pytest.approx(1.9 - 1.5 * h)
    assert sol.v(2) == 0.5 and sol.v(3) == 2 and sol.v(4) == 1.0          # no aperture: the 1 m default radius
    rows = [[float(t) for t in ln.split()] for ln in (tmp_path / "1T2.T7").read_text().splitlines()]
    assert rows[0] == [0, 100, 1] and rows[1][2] == 1503 and rows[1][1] == pytest.approx(100 * (0.3 + 3 * h))
    bz = [rows[2 + 2 * j][1] for j in range(1504)]
    br = [rows[3 + 2 * j][0] for j in range(1504)]
    assert bz[:4] == [0.0, _SOL_EDGE_A, 1 - _SOL_EDGE_A, 1.0] and bz[-4:] == [1.0, 1 - _SOL_EDGE_A, _SOL_EDGE_A, 0.0]
    assert br[:4] == pytest.approx([0.0, -0.25 / h, -0.25 / h, 0.0])
    assert br[-4:] == pytest.approx([0.0, 0.25 / h, 0.25 / h, 0.0])
    assert all(v == 1.0 for v in bz[4:-4]) and all(v == 0.0 for v in br[4:-4])
    i1, i2, zlen = solenoid_table_integrals(rows)
    assert i1 == pytest.approx(0.3, abs=1e-12) and zlen == pytest.approx(0.3 + 3 * h)   # ∫Bz exact per end
    assert i2 > 0.3                                                             # the overshoot: ∫Bz² > L


def test_rf_profile_and_its_calibration():
    z, ez = raised_cosine(0.2, 0.15, 201)
    coefs = fourier_coefficients(z, ez, 0.2, 40)
    assert len(coefs) == 81 and coefs[0] == pytest.approx(2 * 0.15 / 0.2)       # a0 = (2/L)∫Ez, peak 2
    assert max(abs(evaluate_fourier(coefs, zz, 0.2) - e) for zz, e in zip(z, ez, strict=True)) < 2e-3 * max(ez)
    mass = 938272088.16
    for charge, phi, v0 in ((1.0, -30.0, 1e6), (-1.0, -30.0, 1e6), (1.0, 0.0, 3e5), (1.0, 40.0, 3e5)):
        scale, theta = calibrate(coefs, 0.2, v0, math.radians(phi), 162.5e6, 1e-8, 2.1e6, mass, charge)
        de, v, psi = gain_from_profile(coefs, 0.2, scale, theta, 162.5e6, 1e-8, 2.1e6, mass, charge)
        assert v == pytest.approx(v0, rel=1e-9) and psi == pytest.approx(math.radians(phi), abs=1e-7)
        assert de == pytest.approx(v0 * math.cos(math.radians(phi)), rel=1e-9)
        ke_out, t_out = integrate_reference(coefs, 0.2, scale, theta, 162.5e6, 1e-8, 2.1e6, mass, charge)
        assert ke_out - 2.1e6 == pytest.approx(de, rel=1e-12) and t_out > 1e-8 + 0.2 / 3e8


def test_thin_gap_is_absorbed_by_neighbouring_drifts(tmp_path):
    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", length=0.0,
                                               rf=RFP(voltage_V=1e5, phase_rad=-0.5, frequency_Hz=162.5e6)),
                                      Drift(name="d2", length=0.5)], ref_at())
    _, rep, _, cards = _deck(lat, tmp_path)
    by = _by_name(cards)
    L = by["c"][0].length                                       # the balanced surrogate, split between the drifts
    assert 0.001 < L < 0.5 * lat.reference.beta * 299792458.0 / 162.5e6
    assert by["d1"][0].length == pytest.approx(0.5 - L / 2) and by["d2"][0].length == pytest.approx(0.5 - L / 2)
    assert by["c"][0].zedge == pytest.approx(0.5 - L / 2)
    assert cards[-1].zedge == pytest.approx(1.0) and "THIN_GAP_ADDS_LENGTH" not in rep.codes()
    assert rep.codes()["THIN_GAP_AS_SHORT_CAVITY"] == 1
    _, _, _, cards = _deck(lat, tmp_path / "fixed", thin_gap_length_m=0.001)
    assert _by_name(cards)["c"][0].length == 0.001


def test_negative_species_keeps_the_gain(tmp_path):
    lat = all_kinds_lattice("h-")
    out, _, header, _ = _deck(lat, tmp_path)
    assert header.charge == -1
    back, _ = read(out)
    assert back.reference.species.charge == -1
    assert back.elements["c2"].rf.voltage_V == pytest.approx(1e6, rel=1e-6)
    assert back.elements["c2"].rf.phase_rad == pytest.approx(math.radians(-30.0), abs=1e-6)


def test_strict_mode_raises_on_a_lossy_conversion(tmp_path):
    lat = all_kinds_lattice()                                       # THIN_GAP_ADDS_LENGTH is lossy
    with pytest.raises(TranslationError):
        write(lat, tmp_path / "ImpactT.in", "impactt", strict=True)
    lat2 = Lattice.from_sequence("s", [Sextupole(name="s", length=0.1, multipole=MagneticMultipoleP(Bn={2: 3.0}))],
                                 ref_at())
    write(lat2, tmp_path / "s" / "ImpactT.in", "impactt", strict=True)     # type 5 is exact


def test_no_rf_frequency_is_recorded(tmp_path):
    lat = Lattice.from_sequence("q", [Drift(name="d", length=0.2),
                                      Quadrupole(name="q", length=0.2, multipole=MagneticMultipoleP(Bn={1: 1.0}))],
                                 ref_at(f=None))
    _, rep, header, _ = _deck(lat, tmp_path)
    assert header.frequency_Hz == 1e9 and "IMPACTT_NO_FREQUENCY" in rep.codes()


def test_goldens(tmp_path):
    lat, _ = read(DATA / "helix" / "mebt_line.dat", "tracewin", species="h-", kinetic_energy_eV=2.1e6,
                  frequency_Hz=162.5e6)
    out = tmp_path / "mebt" / "ImpactT.in"
    write(lat, out, "impactt")
    _golden("mebt_line.impactt.in", out)
    out2 = tmp_path / "all" / "ImpactT.in"
    write(all_kinds_lattice(), out2, "impactt")
    _golden("all_kinds.impactt.in", out2)
    _golden("all_kinds.rfdata1", out2.parent / "rfdata1")
