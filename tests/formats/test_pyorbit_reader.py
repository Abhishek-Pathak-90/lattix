"""PyORBIT3 linac XML reader: the public SNS/ESS decks, round trips, fixed points, conventions."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import compare_profiles, neutral_set, profile
from lattix.formats.base import guess_format
from tests.formats.test_pyorbit_writer import all_kinds_lattice

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "pyorbit3"


def _cycle(lat, tmp_path, **opts):
    out = tmp_path / "a.pyorbit.xml"
    rep = write(lat, out, "pyorbit")
    back, rep2 = read(out, **opts)
    return out, rep, back, rep2


@pytest.mark.parametrize("sp", ["proton", "h-"])
def test_all_kinds_round_trip_and_fixed_point(tmp_path, sp):
    lat = all_kinds_lattice(sp)
    out, rep, back, rep2 = _cycle(lat, tmp_path)
    assert back.reference.species.name.lower() == sp and "REFERENCE_FROM_TAG" in rep2.codes()
    diff = compare_profiles(profile(lat, neutral_set(rep)), profile(back, neutral_set(rep)))
    assert diff.ok, diff.problems[:5]
    kinds = Counter(e.kind for e in back.elements.values())
    assert kinds["Quadrupole"] == 1 and kinds["Bend"] == 1 and kinds["Solenoid"] == 1 and kinds["RFCavity"] == 2
    assert kinds["Kicker"] == 2                                # the kicker and the thin multipole's dipole terms
    c1 = back.elements["c1"]
    assert c1.rf.phase_rad == pytest.approx(math.radians(-85.0)) and c1.rf.voltage_V == pytest.approx(8e4)
    assert c1.rf.frequency_Hz == pytest.approx(162.5e6) and back.elements["c2"].rf.frequency_Hz == pytest.approx(325e6)
    k = back.elements["k1"]
    assert k.hkick == pytest.approx(1e-3) and k.vkick == pytest.approx(-2e-3)
    assert k.length == pytest.approx(0.1) and back.elements["c2"].length == pytest.approx(0.2)   # markers folded
    assert rep2.codes().get("PYORBIT_END_MARKERS_FOLDED") == 2 and "k1_in" not in back.elements
    assert back.elements["sol1"].solenoid.Bsol_T == pytest.approx(0.5)
    out2 = tmp_path / "b.pyorbit.xml"
    write(back, out2, "pyorbit")
    assert out.read_text() == out2.read_text()


@pytest.mark.parametrize("name, sp, ke, n_gaps, n_split", [
    ("sns_mebt.xml", "h-", 2.5e6, 4, 18), ("ess_mebt.xml", "proton", 3.6e6, 3, 31),
    ("solenoid_test.xml", "proton", 3.6e6, 0, 0)])
def test_public_decks_read_and_round_trip(tmp_path, name, sp, ke, n_gaps, n_split):
    lat, rep = read(DATA / name, species=sp, kinetic_energy_eV=ke)
    kinds = Counter(e.kind for e in lat.elements.values())
    assert kinds["RFCavity"] == n_gaps
    assert rep.codes().get("PYORBIT_THIN_NODE_SPLITS_MAGNET", 0) == n_split
    assert all(e.length >= 0 for e in lat.elements.values())
    out = tmp_path / (name + ".a.pyorbit.xml")
    write(lat, out, "pyorbit")
    back, _ = read(out)
    diff = compare_profiles(profile(lat, {}), profile(back, {}))
    assert diff.ok, diff.problems[:5]
    out2 = tmp_path / (name + ".b.pyorbit.xml")
    write(back, out2, "pyorbit")
    assert out.read_text() == out2.read_text()


def test_sns_mebt_conventions():
    lat, _ = read(DATA / "sns_mebt.xml", species="h-", kinetic_energy_eV=2.5e6)
    assert sum(p.element.length for p in lat.flatten()) == pytest.approx(3.633)
    q = next(e for e in lat.elements.values() if e.kind == "Quadrupole" and e.name.startswith("MEBT_Mag:QH01"))
    assert q.multipole.Bn[1] == pytest.approx(-34.636)                      # the lab gradient, T/m
    assert q.aperture is not None and q.aperture.x_limits == pytest.approx((-0.016, 0.016))
    gap = next(e for e in lat.elements.values() if e.kind == "RFCavity")
    assert gap.rf.frequency_Hz == pytest.approx(402.5e6) and gap.rf.voltage_V == pytest.approx(7.5e-5 * 1e9)
    assert gap.rf.phase_rad == pytest.approx(math.radians(90.0 + 180.0) - 2 * math.pi)   # H⁻: phase + 180°
    assert "ttfs_xml" in gap.native["pyorbit"]
    lat2, _ = read(DATA / "sns_mebt.xml", species="h-", kinetic_energy_eV=2.5e6, frequency_Hz=402.5e6)
    assert lat2.reference.rf_frequency_Hz == 402.5e6


def test_reference_from_tag_or_options(tmp_path):
    lat = all_kinds_lattice()
    out, _, back, rep2 = _cycle(lat, tmp_path)
    assert "REFERENCE_FROM_TAG" in rep2.codes()
    text = out.read_text().replace("# lattix: reference", "# no reference")
    (tmp_path / "notag.pyorbit.xml").write_text(text)
    with pytest.raises(ValueError, match="species"):
        read(tmp_path / "notag.pyorbit.xml")
    lat2, _ = read(tmp_path / "notag.pyorbit.xml", species="proton", kinetic_energy_eV=2.1e6)
    assert lat2.reference.rf_frequency_Hz == pytest.approx(162.5e6)      # bpmFrequency


def test_bare_xml_is_the_pyorbit_format():
    assert guess_format(DATA / "sns_mebt.xml") == "pyorbit"
