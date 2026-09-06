"""OPAL-T reader: round trips and fixed points, a hand-written native deck (inheritance, variables, nested
LINEs, ELEMEDGE gaps, DESIGNENERGY / map-integrated voltages), reference options, sniffing."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.corpus import sniff_text
from lattix.crossval import compare_profiles, neutral_set, profile
from lattix.formats.base import guess_format
from lattix.formats.impactt.rfprofile import transit_factor
from lattix.formats.opal.maps import dynamic_map_text, plateau, raised_bump, static_map_text
from lattix.ir.lattice import Lattice
from tests.formats.test_ocelot_writer import all_kinds_lattice

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
C = 299_792_458.0


def _cycle(lat: Lattice, d: Path, i: int):
    out = d / f"c{i}" / "a.opal.in"                 # one directory per cycle: the map files keep their names
    rep = write(lat, out, "opal")
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
            "Instrument", "Foil", "Patch", "ReferenceChange", "Freq", "Multipole", "Marker", "Taylor",
            "Directive"} <= kinds
    thin = back1.elements["c1"]
    assert thin.length == 0.0 and thin.rf.voltage_V == 8e4 and thin.rf.phase_rad == pytest.approx(math.radians(-85))
    assert back1.elements["sol1"].length == 0.3 and back1.elements["sol1"].solenoid.Bsol_T == pytest.approx(0.5)
    assert back1.elements["m1"].multipole.BnL == pytest.approx({0: 0.01, 1: 0.02}) and back1.elements["m1"].length == 0
    assert back1.elements["k1"].hkick == 1e-3 and back1.elements["col1"].aperture.half_y == 0.02
    assert [p.s_in for p in back1.flatten()] == pytest.approx([p.s_in for p in lat.flatten()])
    out2, _, back2, _ = _cycle(back1, tmp_path, 2)
    out3, _, _, _ = _cycle(back2, tmp_path, 3)
    assert out2.read_text() == out3.read_text()
    assert (out2.parent / "a_c2.1dd").read_text() == (out3.parent / "a_c2.1dd").read_text()


@pytest.mark.parametrize("rel,opts", [("helix/fodo.madx", {}), ("helix/mebt_line.dat", {}),
                                      ("flame/LS1.lat", {}), ("pals/rf_voltage.pals.yaml", {})])
def test_public_decks_round_trip(tmp_path, rel, opts):
    lat, _ = read(DATA / rel, **opts)
    out, rep, back, rep2 = _cycle(lat, tmp_path, 1)
    diff = compare_profiles(profile(lat, neutral_set(rep)), profile(back, neutral_set(rep)))
    assert diff.ok, diff.problems[:5]
    assert back.flatten()[-1].s_out == pytest.approx(lat.flatten()[-1].s_out)


NATIVE = '''
// a native OPAL-T deck (no lattix tags)
Title, string="native injector";
OPTION, AUTOPHASE=4, ECHO=FALSE;
REAL kq = 0.5;
REAL lq = 2 * 0.1;
d1: DRIFT, L=0.3, ELEMEDGE=0.0;
qf: QUADRUPOLE, L=lq, K1=kq, ELEMEDGE=0.3, APERTURE="CIRCLE(0.04)";
qd: qf, K1=-kq, ELEMEDGE=0.5, DX=1.0e-3;      /* inherits from qf */
b1: RBEND, L=0.5, ANGLE=0.1, E1=0.0, DESIGNENERGY=2.1, ELEMEDGE=1.0;
c1: RFCAVITY, L=0.1, VOLT=5.0, FREQ=162.5, LAG=-0.5236, DESIGNENERGY=2.6, FMAPFN="c1.1dd", ELEMEDGE=1.5;
c2: RFCAVITY, L=0.1, VOLT=5.0, FREQ=162.5, LAG=0.0, FMAPFN="c1.1dd", ELEMEDGE=1.6;
sol: SOLENOID, L=0.2, KS=2.0, FMAPFN="sol.1dms", ELEMEDGE=1.7;
k1: HKICKER, L=0.05, KICK=1.0e-3, ELEMEDGE=1.9;
col: RCOLLIMATOR, L=0.02, XSIZE=0.01, YSIZE=0.02, ELEMEDGE=1.95;
bpm: MONITOR, ELEMEDGE=1.97;
src: SOURCE, ELEMEDGE=1.97;
m: MARKER, ELEMEDGE=1.97;
cell: LINE = (d1, qf, qd, b1);
all: LINE = (cell, c1, c2, sol, k1, col, bpm, src, m);
bm: BEAM, PARTICLE=PROTON, PC=0.0628103, NPART=100;
fs: FIELDSOLVER, FSTYPE=NONE, MX=8, MY=8, MT=8;
ds: DISTRIBUTION, TYPE=GAUSS, SIGMAX=1e-3;
TRACK, LINE=all, BEAM=bm, MAXSTEPS=1000, DT=1e-12, ZSTOP=2.0;
RUN, METHOD="PARALLEL-T", BEAM=bm, FIELDSOLVER=fs, DISTRIBUTION=ds;
ENDTRACK;
QUIT;
'''


def _native(tmp_path: Path) -> Path:
    z, ez = raised_bump(0.1, 0.06, 101)
    (tmp_path / "c1.1dd").write_text(dynamic_map_text(z, ez, 162.5e6))
    z, bz = plateau(0.2, 0.0, 41)
    (tmp_path / "sol.1dms").write_text(static_map_text(z, bz))
    p = tmp_path / "native.opal.in"
    p.write_text(NATIVE)
    return p


def test_native_deck(tmp_path):
    p = _native(tmp_path)
    assert guess_format(p) == "opal" and sniff_text(NATIVE, ".in") == "opal"
    lat, rep = read(p)
    ref = lat.reference
    assert ref.species.name == "proton" and ref.kinetic_energy_eV == pytest.approx(2.1e6, rel=1e-4)
    seq = [p.element for p in lat.flatten()]
    names = [e.name for e in seq]
    assert names == ["d1", "qf", "qd", "b1_gap", "b1", "c1", "c2", "sol", "k1", "col", "bpm", "src", "m"]
    assert seq[1].gradient == pytest.approx(0.5 * ref.brho_abs)
    assert seq[2].gradient == pytest.approx(-0.5 * ref.brho_abs)
    assert seq[1].length == pytest.approx(0.2) and seq[1].aperture.half_x == 0.02 and seq[2].shift.x_offset == 1e-3
    assert seq[3].kind == "Drift" and seq[3].length == pytest.approx(0.3)          # the ELEMEDGE gap
    b = seq[4]
    assert b.kind == "Bend" and b.bend.rect and b.bend.e1 == pytest.approx(0.05) and b.bend.e2 == pytest.approx(0.05)
    c1, c2 = seq[5], seq[6]
    assert c1.rf.voltage_V == pytest.approx(2.6e6 - ref.kinetic_energy_eV, rel=1e-9)   # the crest energy
    assert c1.rf.phase_rad == pytest.approx(-0.5236) and c1.rf.frequency_Hz == 162.5e6 and c1.length == 0.1
    # c2 has no DESIGNENERGY: its voltage is the crest gain of the map at VOLT (5 MV/m), close to the
    # first-order transit-time estimate
    ref2 = ref.advanced(dE_eV=c1.rf.voltage_V * math.cos(-0.5236))
    z, ez = raised_bump(0.1, 0.06, 101)
    est = 5e6 * abs(transit_factor(z, [v / max(ez) for v in ez], 2 * math.pi * 162.5e6 / (ref2.beta * C)))
    assert c2.rf.voltage_V == pytest.approx(est, rel=0.05) and c2.rf.phase_rad == 0.0
    assert seq[7].solenoid.Bsol_T == pytest.approx(2.0 * ref.brho_abs) and seq[7].length == pytest.approx(0.2)
    assert seq[8].hkick == 1e-3 and seq[8].length == 0.05
    assert seq[9].aperture.half_x == 0.01 and seq[9].aperture.half_y == 0.02 and seq[9].aperture.shape == "RECTANGULAR"
    assert seq[10].kind == "Instrument" and seq[11].kind == "Marker" and seq[12].kind == "Marker"
    codes = {e.code for e in rep.entries}
    assert {"OPAL_GAP_DRIFT_INSERTED", "RBEND_AS_SECTOR", "OPAL_LAG_AS_SYNC_PHASE", "UNSUPPORTED_OPAL_ELEMENT"} <= codes
    assert lat.name == "native injector" and lat.use == "ALL"
    # a native deck re-written keeps its maps
    out = tmp_path / "again" / "n.opal.in"
    write(lat, out, "opal")
    again, _ = read(out)
    assert [e.name for e in (p.element for p in again.flatten())] == names
    assert again.elements["c1"].rf.voltage_V == pytest.approx(c1.rf.voltage_V)


def test_reference_options_and_ion(tmp_path):
    p = _native(tmp_path)
    lat, _ = read(p, species="h-", kinetic_energy_eV=3e6, frequency_Hz=325e6)
    assert lat.reference.species.name == "h-" and lat.reference.kinetic_energy_eV == 3e6
    assert lat.reference.rf_frequency_Hz == 325e6
    text = NATIVE.replace("PARTICLE=PROTON, PC=0.0628103", "PARTICLE=PROTON, MASS=1.8756, CHARGE=1, ENERGY=1.8776")
    q = tmp_path / "ion.opal.in"
    q.write_text(text)
    lat, rep = read(q)
    assert lat.reference.species.mass_eV == pytest.approx(1.8756e9)
    assert "SPECIES_ASSUMED" in {e.code for e in rep.entries}
    assert lat.reference.kinetic_energy_eV == pytest.approx(2.0e6, rel=1e-9)


def test_strict_and_line_selection(tmp_path):
    p = _native(tmp_path)
    lat, _ = read(p, line="cell")
    assert [e.name for e in (x.element for x in lat.flatten())] == ["d1", "qf", "qd", "b1_gap", "b1"]
    from lattix.fidelity import TranslationError
    with pytest.raises(TranslationError):
        read(p, strict=True)
