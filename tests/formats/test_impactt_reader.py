"""IMPACT-T reader: the vendored Sample1, the all-kinds fixed point, gaps/overlaps/thin-inside-thick."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import compare_profiles, neutral_set, profile
from lattix.fidelity import TranslationError
from lattix.formats.base import guess_format
from lattix.formats.impactt.reader import read_rfdata
from lattix.ir.elements import Drift, MagneticMultipoleP, Quadrupole
from lattix.ir.lattice import Lattice
from tests.formats.test_pyorbit_writer import all_kinds_lattice, ref_at

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "impactt"
HELIX = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"


@pytest.mark.parametrize("sp", ["proton", "h-"])
def test_all_kinds_round_trip_and_fixed_point(tmp_path, sp):
    lat = all_kinds_lattice(sp)
    out = tmp_path / "a" / "ImpactT.in"
    rep = write(lat, out, "impactt")
    back, rep2 = read(out)
    assert back.reference.species.name.lower() == sp and back.reference.kinetic_energy_eV == 2.1e6
    assert back.reference.rf_frequency_Hz == 162.5e6 and back.name == "all_kinds"
    assert "THIN_GAP_ADDS_LENGTH" in rep.codes()          # c1 sits between a solenoid and a cavity: +1 mm
    kinds = Counter(e.kind for e in back.elements.values())
    assert kinds == {"Drift": 3, "Quadrupole": 1, "Sextupole": 1, "Octupole": 1, "Kicker": 2, "Bend": 1,
                     "Solenoid": 1, "RFCavity": 2, "Collimator": 1, "Marker": 1, "Instrument": 1, "Foil": 1,
                     "Taylor": 1, "Patch": 1}                    # c1 back as a thin gap, its 1 mm an implicit drift
    assert back.elements["c1"].length == 0.0 and back.elements["c1"].meta["impactt_thin"]["pad"] == 0.0
    assert back.elements["q1"].multipole.Bn[1] == 1.2 and back.elements["s1"].multipole.Bn[2] == 3.0
    b = back.elements["b1"]
    assert b.bend.angle == pytest.approx(0.1) and b.bend.e1 == pytest.approx(0.05) and b.bend.e2 == pytest.approx(0.05)
    sol = back.elements["sol1"]
    assert sol.length == pytest.approx(0.3) and sol.solenoid.Bsol_T == 0.5 and "impactt_table" in sol.meta
    for name, v, phi in (("c1", 8e4, -85.0), ("c2", 1e6, -30.0)):
        c = back.elements[name]
        assert c.rf.voltage_V == pytest.approx(v, rel=1e-6)
        assert c.rf.phase_rad == pytest.approx(math.radians(phi), abs=1e-6)
    assert back.elements["c1"].rf.frequency_Hz == 162.5e6 and back.elements["c2"].rf.frequency_Hz == 325e6
    k = back.elements["k1"]
    assert k.hkick == pytest.approx(1e-3) and k.vkick == pytest.approx(-2e-3) and k.length == pytest.approx(0.1)
    col = back.elements["col1"]
    assert col.length == pytest.approx(0.05) and col.aperture.x_limits == pytest.approx((-0.01, 0.01))
    assert rep2.codes() == {"IMPACTT_RF_DRIVEN_PHASE": 2, "SIM_SETTINGS_KEPT_IN_META": 1}
    out2 = tmp_path / "b" / "ImpactT.in"
    write(back, out2, "impactt")
    assert out.read_text() == out2.read_text()
    for f in ("rfdata1", "1T2.T7", "rfdata3", "rfdata4"):
        assert (out.parent / f).read_text() == (out2.parent / f).read_text()


def test_sample1_reads_and_is_a_fixed_point(tmp_path):
    lat, rep = read(DATA / "Sample1" / "ImpactT.in")
    h = lat.meta["impactt_header"]
    assert h["np"] == 10000 and h["flagdist"] == 112 and h["current_A"] == 1.3 and h["nemission"] == 35
    assert lat.reference.species.mass_eV == 511005.0 and lat.reference.species.charge == -1
    assert lat.reference.kinetic_energy_eV == 0.5 and lat.reference.rf_frequency_Hz == 1.3e9
    kinds = Counter(e.kind for e in lat.elements.values())
    assert kinds == {"RFCavity": 3, "NCells": 1, "Drift": 1}          # the overlaps leave one gap
    assert rep.codes() == {"IMPACTT_RF_GAIN_UNKNOWN": 4, "IMPACTT_SOLRF_BZ_DROPPED": 3, "IMPACTT_OVERLAP": 4,
                           "SIM_SETTINGS_KEPT_IN_META": 1, "IMPACTT_BEAM_CURRENT": 1}
    first = next(iter(lat.elements.values()))
    assert first.native["impactt"]["type"] == 105 and first.native["impactt"]["passthrough"] and first.length == 2.0
    out = tmp_path / "ImpactT.in"
    rep_w = write(lat, out, "impactt")
    assert rep_w.codes() == {"IMPACTT_NATIVE_POSITION": 1}          # the overlapping cards keep their zedge
    back, _ = read(out)
    out2 = tmp_path / "b" / "ImpactT.in"
    write(back, out2, "impactt")
    assert out.read_text() == out2.read_text()
    for f in ("rfdata1", "rfdata2", "rfdata3"):          # IMPACT-T reads the first number of every line
        assert read_rfdata(out.parent / f) == pytest.approx(read_rfdata(DATA / "Sample1" / f), rel=1e-12)


def test_implicit_gaps_overlaps_and_thin_inside_thick(tmp_path):
    lat = Lattice.from_sequence("t", [Drift(name="d1", length=0.5),
                                      Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 1.2})),
                                      Drift(name="d2", length=0.5)], ref_at())
    out = tmp_path / "ImpactT.in"
    write(lat, out, "impactt")
    lines = out.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if "name=d2 " in ln)
    lines[i + 1] = lines[i + 1].replace(" 0 0.7 ", " 0 0.9 ")            # d2 moved: 0.8 … 0.9 m is a gap
    end = next(k for k, ln in enumerate(lines) if " -99 " in ln)
    lines[end:end] = ["0 1 1 -1 0.6 0.6 0 0.0001 0 0 0 0 /",              # a steer inside the quad
                      "0.1 1 1 1 0.65 2 0 0.02 0 0 0 0 0 0 0 /"]          # a quad overlapping it: pushed
    out.write_text("\n".join(lines) + "\n")
    back, rep = read(out)
    codes = rep.codes()
    assert codes["IMPACTT_THIN_INSIDE_THICK"] == 1 and codes["IMPACTT_OVERLAP"] == 1
    seq = [(p.element.kind, round(p.s_in, 6), round(p.element.length, 6)) for p in back.flatten()]
    assert seq == [("Drift", 0.0, 0.5), ("Quadrupole", 0.5, 0.1), ("Kicker", 0.6, 0.0), ("Quadrupole", 0.6, 0.1),
                   ("Quadrupole", 0.7, 0.1), ("Drift", 0.8, 0.1), ("Drift", 0.9, 0.5)]
    kick = next(e for e in back.elements.values() if e.kind == "Kicker")
    assert kick.hkick == pytest.approx(1e-4 / (back.reference.gamma * back.reference.beta))
    pushed = [e for e in back.elements.values() if e.native.get("impactt", {}).get("pushed")]
    assert len(pushed) == 1 and pushed[0].kind == "Quadrupole"
    gap = [e for e in back.elements.values() if e.native.get("impactt", {}).get("implicit")]
    assert len(gap) == 1 and gap[0].kind == "Drift" and gap[0].length == pytest.approx(0.1)
    rep_w = write(back, tmp_path / "again" / "ImpactT.in", "impactt")
    assert any(e.code == "IMPACTT_IMPLICIT_GAP" for e in rep_w.entries) and "IMPACTT_NATIVE_POSITION" in rep_w.codes()
    with pytest.raises(TranslationError):
        read(out, strict=True)


def test_format_is_recognised_by_name():
    assert guess_format(DATA / "Sample1" / "ImpactT.in") == "impactt"


def test_tracewin_mebt_round_trip_is_a_fixed_point(tmp_path):
    """Thin gaps between drifts are absorbed (no length added): the profile survives the round trip."""
    lat, _ = read(HELIX / "mebt_line.dat", "tracewin", species="h-", kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    out = tmp_path / "a" / "ImpactT.in"
    rep = write(lat, out, "impactt")
    assert "THIN_GAP_ADDS_LENGTH" not in rep.codes()
    back, _ = read(out)
    diff = compare_profiles(profile(lat, neutral_set(rep)), profile(back, neutral_set(rep)))
    assert diff.ok, diff.problems[:5]
    out2 = tmp_path / "b" / "ImpactT.in"
    write(back, out2, "impactt")
    assert out.read_text() == out2.read_text()
