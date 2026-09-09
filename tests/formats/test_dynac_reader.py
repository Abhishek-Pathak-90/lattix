"""DYNAC reader: round trips, fixed points, the beam block, hand-written decks, the shipped examples."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.corpus import sniff_text
from lattix.crossval import compare_profiles, neutral_set, profile
from lattix.formats.base import guess_format
from lattix.ir.lattice import Lattice
from lattix.testing import codes_path
from tests.formats.test_ocelot_writer import all_kinds_lattice

DYNAC_CLONE = codes_path('tier3_peers', 'dynac')
SNS_DECK = DYNAC_CLONE / "datafiles" / "sns" / "mebt_dtl1.in"


def _cycle(lat: Lattice, d: Path, i: int):
    out = d / f"c{i}" / "a.dynac.in"                 # one directory per cycle: the side file keeps its name
    rep = write(lat, out, "dynac")
    back, rep2 = read(out)
    return out, rep, back, rep2


@pytest.mark.parametrize("sp", ["proton", "h-"])
def test_all_kinds_round_trip_and_fixed_point(tmp_path, sp):
    lat = all_kinds_lattice(sp)
    out1, rep, back1, rep2 = _cycle(lat, tmp_path, 1)
    assert back1.reference.species.name.lower() == sp and back1.reference.kinetic_energy_eV == 2.1e6
    assert back1.reference.rf_frequency_Hz == 162.5e6
    diff = compare_profiles(profile(lat, neutral_set(rep)), profile(back1, neutral_set(rep)))
    assert diff.ok, diff.problems[:5]
    kinds = {e.kind for e in back1.elements.values()}
    assert {"Quadrupole", "Sextupole", "Octupole", "Bend", "Solenoid", "RFCavity", "Kicker", "Collimator",
            "Instrument", "Foil", "Patch", "ReferenceChange", "Freq", "Multipole", "Marker"} <= kinds
    out2, _, back2, _ = _cycle(back1, tmp_path, 2)
    out3, _, _, _ = _cycle(back2, tmp_path, 3)
    assert out2.read_text(encoding="latin-1") == out3.read_text(encoding="latin-1")


def test_restored_details(tmp_path):
    lat = all_kinds_lattice()
    _, _, back, rep2 = _cycle(lat, tmp_path, 1)
    e = back.elements
    assert e["q1"].multipole.Bn[1] == pytest.approx(1.2) and e["q1"].shift.x_offset == pytest.approx(1e-3)
    assert e["q1"].aperture.x_limits == pytest.approx((-0.05, 0.05))
    assert e["s1"].multipole.Bn[2] == pytest.approx(3.0) and e["sol1"].solenoid.Bsol_T == pytest.approx(0.5)
    b = e["b1"].bend
    assert b.angle == pytest.approx(0.1) and b.e1 == pytest.approx(0.05) and b.edge_int1 == 0.45
    assert b.hgap == pytest.approx(0.03) and e["b1"].length == pytest.approx(1.0)
    c1, c2 = e["c1"], e["c2"]
    assert c1.length == 0.0 and c1.rf.voltage_V == pytest.approx(8e4)
    assert c1.rf.phase_rad == pytest.approx(math.radians(-85.0)) and c1.rf.frequency_Hz == 162.5e6
    assert c2.length == pytest.approx(0.2) and c2.rf.voltage_V == pytest.approx(1e6)
    assert c2.rf.phase_rad == pytest.approx(math.radians(-30.0)) and c2.rf.cavity_type == "TRAVELING_WAVE"
    k = e["k1"]
    assert k.hkick == pytest.approx(1e-3) and k.vkick == pytest.approx(-2e-3) and k.length == pytest.approx(0.1)
    m = e["m1"].multipole
    assert m.BnL[0] == pytest.approx(0.01) and m.BsL[0] == pytest.approx(0.02)
    assert e["col1"].length == pytest.approx(0.05) and e["col1"].aperture.x_limits == pytest.approx((-0.01, 0.01))
    assert e["col1"].aperture.shape == "RECTANGULAR" and e["d4"].aperture.shape == "ELLIPTICAL"
    assert e["f1"].material == "C" and e["f1"].thickness_kg_per_m2 == pytest.approx(1e-3)
    assert e["p1"].x_offset == pytest.approx(1e-3) and e["rc1"].dE_ref_eV == 1e3
    assert e["fq1"].frequency_Hz == 162.5e6 and e["bpm1"].family == "BPM" and e["scr1"].family == "SCREEN"
    assert "REFERENCE_FROM_TAG" in rep2.codes()


def test_hand_written_deck(tmp_path):
    text = """SNS-LIKE MEBT PIECE
GEBEAM
2 1
402.5E06
1000
0. 0.0 0.  0.0 0. 0.
-1.96154 0.18309 10.91
 1.76906 0.16116 10.93
 0.0179  0.77283 512.83
INPUT
939.301404 1. -1.
2.50     0.
EMIPRT
0
SCDYNAC
3
38. 3.
0
DRIFT
  9.75
QUADRUPO
  6.1  -5.1954957  1.5
DRIFT
  8.4
BUNCHER
  -0.075 -90. 1  1.5
STEER
0.002 1
NEWF
805.E06
BUNCHER
  0.05 120. 1 1.5
EMIT
STOP
"""
    p = tmp_path / "mebt.dynac.in"
    p.write_text(text)
    assert guess_format(p) == "dynac" and sniff_text(text, "mebt.in") == "dynac"
    lat, rep = read(p)
    ref = lat.reference
    assert ref.species.name == "h-" and ref.species.charge == -1 and ref.kinetic_energy_eV == pytest.approx(2.5e6)
    assert ref.rf_frequency_Hz == pytest.approx(402.5e6)
    seq = [pl.element for pl in lat.flatten()]
    kinds = [e.kind for e in seq]
    assert kinds == ["Directive", "Drift", "Quadrupole", "Drift", "RFCavity", "Kicker", "Freq", "RFCavity", "Marker"]
    q = seq[2]
    assert q.length == pytest.approx(0.061) and q.multipole.Bn[1] == pytest.approx(10.0 * -5.1954957 / 1.5)
    assert q.aperture.x_limits == pytest.approx((-0.015, 0.015))
    b1 = seq[4]
    assert b1.rf.voltage_V == pytest.approx(-7.5e4) and b1.rf.frequency_Hz == pytest.approx(402.5e6)
    assert b1.rf.phase_rad == pytest.approx(math.radians(90.0))          # −90° − 180° (H⁻), wrapped
    assert seq[5].vkick == pytest.approx(0.002 / ref.brho_signed) and ref.brho_signed < 0
    assert seq[6].frequency_Hz == 805e6 and seq[7].rf.frequency_Hz == pytest.approx(805e6)
    assert seq[0].format == "dynac" and seq[0].card == "SCDYNAC" and seq[0].args == ["3", "38. 3.", "0"]
    out = tmp_path / "again.dynac.in"
    rep2 = write(lat, out, "dynac")
    assert "DYNAC_DIRECTIVE_KEPT" in [e.code for e in rep2.entries]
    assert "SCDYNAC\n3\n38. 3.\n0\n" in out.read_text(encoding="latin-1")


def test_deck_without_beam_needs_options(tmp_path):
    p = tmp_path / "bare.dynac.in"
    p.write_text("BARE\nDRIFT\n10.\nSTOP\n")
    with pytest.raises(ValueError, match="species"):
        read(p)
    lat, _ = read(p, species="proton", kinetic_energy_eV=3e6, frequency_Hz=1e8)
    assert lat.total_length == pytest.approx(0.1) and lat.reference.kinetic_energy_eV == 3e6


@pytest.mark.skipif(not SNS_DECK.is_file(), reason="DYNAC clone with the SNS example not present")
def test_shipped_sns_deck_reads(tmp_path):
    lat, rep = read(SNS_DECK, "dynac")
    kinds = [pl.element.kind for pl in lat.flatten()]
    assert kinds.count("Quadrupole") == 77 and kinds.count("RFCavity") == 64      # 4 bunchers + 60 DTL gaps
    assert lat.reference.species.name == "h-" and lat.reference.kinetic_energy_eV == pytest.approx(2.5e6)
    assert "CAVSC_AS_GAP" in rep.codes()
    assert lat.total_length > 7.0
    end = lat.flatten()[-1]
    from lattix.ir.walk import propagate

    w_end = propagate(lat)[-1].ref_out.kinetic_energy_eV
    assert 7.0e6 < w_end < 8.0e6                     # DYNAC's own run ends at 7.548 MeV (dynac.print.ref)
    out = tmp_path / "sns_again.dynac.in"
    rep2 = write(lat, out, "dynac")
    assert rep2.ok and end is not None
