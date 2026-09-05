"""Ocelot reader: round trips, fixed points, the reference, folding, hand-written modules."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import compare_profiles, neutral_set, profile
from lattix.formats.base import guess_format
from lattix.ir.elements import Drift
from lattix.ir.lattice import Lattice
from tests.formats.test_ocelot_writer import all_kinds_lattice, ref_at

M_E = 510998.95


def _cycle(lat: Lattice, tmp_path: Path, **read_opts):
    out = tmp_path / "a.ocelot.py"
    rep = write(lat, out, "ocelot")
    back, rep2 = read(out, **read_opts)
    return out, rep, back, rep2


@pytest.mark.parametrize("sp", ["electron", "proton"])
def test_all_kinds_round_trip_and_fixed_point(tmp_path, sp):
    lat = all_kinds_lattice(sp)
    out, rep, back, rep2 = _cycle(lat, tmp_path)
    assert back.reference.species.name.lower() == sp and back.reference.kinetic_energy_eV == 2.1e6
    diff = compare_profiles(profile(lat, neutral_set(rep)), profile(back, neutral_set(rep)))
    assert diff.ok, diff.problems[:5]
    kinds = {e.kind for e in back.elements.values()}
    assert {"Quadrupole", "Sextupole", "Octupole", "Bend", "Solenoid", "RFCavity", "Kicker", "Collimator",
            "Instrument", "Foil", "Taylor", "Patch", "ReferenceChange", "Freq", "Directive", "Multipole",
            "Marker"} <= kinds
    out2 = tmp_path / "b.ocelot.py"
    write(back, out2, "ocelot")
    assert out.read_text() == out2.read_text()


def test_restored_details(tmp_path):
    lat = all_kinds_lattice()
    _, _, back, _ = _cycle(lat, tmp_path)
    brho = lat.reference.brho_signed
    assert back.elements["q1"].multipole.Bn[1] == pytest.approx(1.2) and back.elements["q1"].shift.x_offset == 1e-3
    assert back.elements["sol1"].solenoid.Bsol_T == pytest.approx(0.5)
    b = back.elements["b1"].bend
    assert b.hgap == pytest.approx(0.03) and b.edge_int1 == 0.45 and b.e1 == 0.05 and not b.rect
    c1, c2 = back.elements["c1"], back.elements["c2"]
    assert c1.length == 0.0 and c1.rf.voltage_V == pytest.approx(8e4)
    assert c1.rf.phase_rad == pytest.approx(math.radians(-85.0))
    assert c2.rf.cavity_type == "TRAVELING_WAVE" and c2.rf.voltage_V == pytest.approx(1e6)
    k = back.elements["k1"]
    assert k.kind == "Kicker" and k.hkick == 1e-3 and k.vkick == -2e-3 and k.length == 0.1
    assert "k1_v" not in back.elements
    m = back.elements["m1"].multipole
    assert m.BnL[0] == pytest.approx(0.01) and m.BnL[1] == pytest.approx(0.02) and not m.BsL
    assert back.elements["bpm1"].family == "BPM" and back.elements["scr1"].family == "SCREEN"
    assert back.elements["f1"].material == "C" and back.elements["f1"].thickness_kg_per_m2 == 1e-3
    t = back.elements["t1"]
    assert t.basis == "common" and t.matrix[0][1] == pytest.approx(0.1) and t.matrix[1][1] == 1.0
    assert back.elements["p1"].x_offset == 1e-3 and back.elements["rc1"].dE_ref_eV == 1e3
    assert back.elements["fq1"].frequency_Hz == 162.5e6
    d = back.elements["dir1"]
    assert d.card == "LATTICE" and d.args == ["4", "0"] and d.role == "period_start"
    assert brho < 0


def test_apertures_fold_back_and_the_collimator_body_returns(tmp_path):
    lat = all_kinds_lattice()
    _, _, back, _ = _cycle(lat, tmp_path)
    d4 = back.elements["d4"]
    assert d4.aperture is not None and d4.aperture.aperture_at == "BOTH_ENDS"
    assert d4.aperture.x_limits == pytest.approx((-0.015, 0.015)) and d4.aperture.shape == "ELLIPTICAL"
    col = back.elements["col1"]
    assert col.kind == "Collimator" and col.length == pytest.approx(0.05)
    assert col.aperture.x_limits == pytest.approx((-0.01, 0.01)) and col.aperture.shape == "RECTANGULAR"
    assert "col1_body" not in back.elements and "d4_aper_in" not in back.elements


def test_thin_gap_comes_back_thin_between_its_drifts(tmp_path):
    from lattix.ir.elements import RFP, RFCavity

    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30.0),
                                                                frequency_Hz=162.5e6)),
                                      Drift(name="d2", length=0.5)], ref_at())
    out, _, back, rep2 = _cycle(lat, tmp_path)
    placed = [(p.element.name, p.element.length) for p in back.flatten()]
    assert placed == [("d1", pytest.approx(0.5)), ("c", 0.0), ("d2", pytest.approx(0.5))]
    assert "THIN_GAP_PAD_DRIFT" not in rep2.codes()
    out2 = tmp_path / "b.ocelot.py"
    write(back, out2, "ocelot")
    assert out.read_text() == out2.read_text()


def test_unabsorbed_surrogate_keeps_the_file_survey(tmp_path):
    from lattix.ir.elements import RFP, RFCavity, Solenoid, SolenoidP

    lat = Lattice.from_sequence("g", [Solenoid(name="s", length=0.2, solenoid=SolenoidP(Bsol_T=0.1)),
                                      RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=-0.5, frequency_Hz=1.3e9))],
                                ref_at())
    out, rep, back, rep2 = _cycle(lat, tmp_path)
    assert "THIN_GAP_ADDS_LENGTH" in rep.codes() and "THIN_GAP_PAD_DRIFT" in rep2.codes()
    from tests.formats.test_ocelot_writer import parse_module

    L = parse_module(out.read_text())["c"][1]["l"]
    assert back.total_length == pytest.approx(0.2 + L)
    assert [p.element.kind for p in back.flatten()] == ["Solenoid", "RFCavity", "Drift"]


def test_reference_from_tag_options_or_twiss(tmp_path):
    lat = Lattice.from_sequence("s", [Drift(name="d", length=1.0)], ref_at())
    out, _, back, rep2 = _cycle(lat, tmp_path)
    assert "REFERENCE_FROM_TAG" in rep2.codes()
    text = "\n".join(ln for ln in out.read_text().splitlines() if "lattix: reference" not in ln)
    p = tmp_path / "notag.ocelot.py"
    p.write_text(text)
    lat2, rep3 = read(p)
    assert "REFERENCE_FROM_TWISS" in rep3.codes() and lat2.reference.species.name == "electron"
    assert lat2.reference.kinetic_energy_eV == pytest.approx(2.1e6, rel=1e-9)
    lat3, rep4 = read(p, species="proton", kinetic_energy_eV=5e6)
    assert lat3.reference.species.charge == 1 and lat3.reference.kinetic_energy_eV == 5e6
    assert "REFERENCE_FROM_TWISS" not in rep4.codes()
    text = "\n".join(ln for ln in text.splitlines() if not ln.startswith("tws0"))
    (tmp_path / "nothing.ocelot.py").write_text(text)
    with pytest.raises(ValueError, match="species"):
        read(tmp_path / "nothing.ocelot.py")


def test_hand_written_module(tmp_path):
    text = '''from ocelot import *
import numpy as np
tws0 = Twiss()
tws0.E = 0.130
l_q = 0.2
k = 2 * pi / 10
q = Quadrupole(l=l_q, k1=-k, eid="Q1")
rb = RBend(l=0.8, angle=0.08, e1=0.01, eid="RB")
c = Cavity(0.02, 0.001, 30.0, 1.3e9, eid="C")
u = Undulator(lperiod=0.04, nperiods=10, Kx=1.0, eid="U")
s = Solenoid(l=np.sqrt(0.25), k=1.0, eid="S")
h = Hcor(l=0.1, angle=1e-3, eid="H")
a = Aperture(xmax=0.01, ymax=0.02, dx=0.001, type="ellipt", eid="A")
part = (q, rb)
cell = part + 2 * (c,) + [u, s, h, a]
lat = MagneticLattice(cell)
'''
    p = tmp_path / "hand.py"
    p.write_text(text)
    assert guess_format(p) == "ocelot"
    lat, rep = read(p)
    assert lat.reference.species.name == "electron"
    assert lat.reference.kinetic_energy_eV == pytest.approx(0.130e9 - M_E, rel=1e-9)
    brho = lat.reference.brho_signed
    assert lat.elements["Q1"].multipole.Bn[1] == pytest.approx(-2 * math.pi / 10 * brho)
    rb = lat.elements["RB"]
    assert rb.bend.rect and rb.bend.e1 == pytest.approx(0.05) and rb.bend.e2 == pytest.approx(0.04)
    c = lat.elements["C"]
    assert c.rf.voltage_V == pytest.approx(1e6) and c.rf.phase_rad == pytest.approx(math.radians(-30.0))
    assert c.rf.frequency_Hz == 1.3e9 and c.length == 0.02
    assert lat.elements["U"].kind == "Drift" and lat.elements["U"].length == pytest.approx(0.4)
    assert "UNSUPPORTED_OCELOT_ELEMENT" in rep.codes()
    from lattix.ir.walk import propagate

    brho_s = next(pl.ref_in.brho_signed for pl in propagate(lat) if pl.element.name == "S")
    assert brho_s != brho                                     # the energy follows the two cavities
    assert lat.elements["S"].solenoid.Bsol_T == pytest.approx(2.0 * brho_s) and lat.elements["S"].length == 0.5
    assert lat.elements["H"].hkick == 1e-3
    a = lat.elements["A"]
    assert a.kind == "Collimator" and a.aperture.x_limits == pytest.approx((-0.009, 0.011))
    assert [it.ref for it in lat.lines[lat.use].items] == ["Q1", "RB", "C", "C", "U", "S", "H", "A"]
    assert lat.total_length == pytest.approx(0.2 + 0.8 + 0.04 + 0.4 + 0.5 + 0.1)


def test_code_built_lattice_needs_ocelot(tmp_path):
    p = tmp_path / "loop.py"
    p.write_text("from ocelot import *\ncell = []\nfor i in range(3):\n    cell.append(Drift(l=1.0))\n"
                 "lat = MagneticLattice(cell)\n")
    with pytest.raises(ValueError, match="use_ocelot"):
        read(p, species="electron", kinetic_energy_eV=1e9)


def test_impactx_python_is_not_sniffed_as_ocelot(tmp_path):
    assert guess_format(tmp_path / "x.impactx.py") == "impactx"
    assert guess_format(tmp_path / "x.ocelot.py") == "ocelot" and guess_format(tmp_path / "x.py") == "ocelot"
