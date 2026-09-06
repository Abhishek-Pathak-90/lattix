"""Synergia lattice JSON reader: round trips, fixed points, the reference particle, native archives."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import compare_profiles, neutral_set, profile
from lattix.formats.base import guess_format
from lattix.ir.elements import Drift
from lattix.ir.lattice import Lattice
from tests.formats.test_ocelot_writer import all_kinds_lattice, ref_at

SYNERGIA_CLONE = Path("/Users/abhishekpathak/Desktop/Projects/particle_tracking_codes/tier3_peers/synergia2")
BOOSTER_JSON = SYNERGIA_CLONE / "examples" / "normal_form" / "booster_init_lattice.json"


def _cycle(lat: Lattice, tmp_path: Path, **read_opts):
    out = tmp_path / "a.synergia.json"
    rep = write(lat, out, "synergia")
    back, rep2 = read(out, **read_opts)
    return out, rep, back, rep2


@pytest.mark.parametrize("sp", ["proton", "h-", "electron"])
def test_all_kinds_round_trip_and_fixed_point(tmp_path, sp):
    lat = all_kinds_lattice(sp)
    out, rep, back, rep2 = _cycle(lat, tmp_path)
    assert back.reference.species.name.lower() == sp and back.reference.kinetic_energy_eV == pytest.approx(2.1e6)
    assert back.reference.rf_frequency_Hz == 162.5e6
    diff = compare_profiles(profile(lat, neutral_set(rep)), profile(back, neutral_set(rep)))
    assert diff.ok, diff.problems[:5]
    kinds = {e.kind for e in back.elements.values()}
    assert {"Quadrupole", "Sextupole", "Octupole", "Bend", "Solenoid", "RFCavity", "Kicker", "Collimator",
            "Instrument", "Foil", "Taylor", "Patch", "ReferenceChange", "Freq", "Directive", "Multipole",
            "Marker"} <= kinds
    out2 = tmp_path / "b.synergia.json"
    write(back, out2, "synergia")
    assert json.loads(out.read_text())["value0"] == json.loads(out2.read_text())["value0"]
    assert "ENERGY_MODE_RESTORED" in rep2.codes() and "PHASE_SLIP_RESTORED" not in rep2.codes()


def test_restored_details(tmp_path):
    lat = all_kinds_lattice()
    _, _, back, _ = _cycle(lat, tmp_path)
    e = back.elements
    assert e["q1"].multipole.Bn[1] == pytest.approx(1.2) and e["q1"].shift.x_offset == pytest.approx(1e-3)
    assert e["s1"].multipole.Bn[2] == pytest.approx(3.0) and e["o1"].multipole.Bn[3] == pytest.approx(4.0)
    assert e["sol1"].solenoid.Bsol_T == pytest.approx(0.5)
    b = e["b1"].bend
    assert b.angle == 0.1 and b.e1 == 0.05 and b.edge_int1 == 0.45 and b.hgap == 0.03 and not b.rect
    c1, c2 = e["c1"], e["c2"]
    assert c1.rf.voltage_V == pytest.approx(8e4) and c1.rf.phase_rad == pytest.approx(math.radians(-85.0))
    assert c1.rf.frequency_Hz == pytest.approx(162.5e6) and c2.rf.cavity_type == "TRAVELING_WAVE"
    assert c2.rf.phase_rad == pytest.approx(math.radians(-30.0)) and c2.length == pytest.approx(0.2)
    k = e["k1"]
    assert k.hkick == pytest.approx(1e-3) and k.vkick == pytest.approx(-2e-3) and k.length == 0.1
    m = e["m1"].multipole
    assert m.BnL[0] == pytest.approx(0.01) and m.BnL[1] == pytest.approx(0.02) and m.BsL[0] == pytest.approx(0.02)
    assert e["col1"].aperture.x_limits == pytest.approx((-0.01, 0.01)) and e["col1"].length == 0.05
    assert e["d4"].aperture.shape == "ELLIPTICAL" and e["d4"].aperture.x_limits == pytest.approx((-0.015, 0.015))
    assert e["f1"].material == "C" and e["f1"].thickness_kg_per_m2 == 1e-3
    t = e["t1"]
    assert t.basis == "common" and t.matrix[0][1] == pytest.approx(0.1) and t.matrix[1][1] == 1.0
    assert e["p1"].x_offset == 1e-3 and e["rc1"].dE_ref_eV == 1e3 and e["fq1"].frequency_Hz == 162.5e6
    assert e["bpm1"].family == "BPM" and e["scr1"].family == "SCREEN"


def test_reference_from_archive_or_options(tmp_path):
    lat = Lattice.from_sequence("s", [Drift(name="d", length=1.0)], ref_at(sp="proton"))
    out, _, back, _ = _cycle(lat, tmp_path)
    doc = json.loads(out.read_text())
    doc.pop("lattix")                                     # a native archive: the reference particle itself
    p = tmp_path / "native.synergia.json"
    p.write_text(json.dumps(doc))
    lat2, rep = read(p)
    assert lat2.reference.species.name == "proton" and lat2.reference.kinetic_energy_eV == pytest.approx(2.1e6)
    assert lat2.reference.rf_frequency_Hz is None and "ENERGY_MODE_RESTORED" not in rep.codes()
    lat3, _ = read(p, species="h-", kinetic_energy_eV=3e6, frequency_Hz=1e8)
    assert lat3.reference.species.charge == -1 and lat3.reference.kinetic_energy_eV == 3e6


def test_native_archive_conventions(tmp_path):
    """A hand-made archive with MAD-X's attributes: rbend faces, lag, kicker kick, unknown types."""
    def el(name, stype, doubles=None, vectors=None):
        from lattix.formats.synergia.writer import TYPE_INDEX

        return {"name": name, "format": 1, "stype": stype, "type": TYPE_INDEX.get(stype, 0), "ancestors": [],
                "string_attributes": [],
                "lazy_double_attributes": [{"key": k, "value": {"value0": repr(v)}}
                                           for k, v in (doubles or {}).items()],
                "lazy_vector_attributes": [{"key": k, "value": [{"value0": repr(x)} for x in v]}
                                           for k, v in (vectors or {}).items()],
                "length_attribute_name": "l", "bend_angle_attribute_name": "angle", "revision": 0,
                "markers": {f"value{i}": False for i in range(4)}}
    mass, etot = 0.938272088, 1.738272088
    doc = {"value0": {"name": "ring", "has_reference_particle": True,
                      "reference_particle_value": {"charge": 1, "four_momentum": {"mass": mass, "energy": etot,
                                                                                    "momentum": 1.463, "gamma": 1.85,
                                                                                    "beta": 0.84},
                                                   "state": {f"value{i}": 0.0 for i in range(6)}, "repetition": 0,
                                                   "s": 0.0, "s_n": 0.0, "abs_time": 0.0, "abs_offset": 0.0},
                      "elements": [el("rb", "rbend", {"l": 0.8, "angle": 0.08, "e1": 0.01}),
                                   el("q", "quadrupole", {"l": 0.2, "k1": 5.0, "tilt": 0.1}),
                                   el("c", "rfcavity", {"l": 0.0, "volt": 1.0, "lag": 0.25, "freq": 162.5,
                                                        "harmon": 84.0}),
                                   el("h", "hkicker", {"l": 0.0, "kick": 1e-3}),
                                   el("m", "multipole", vectors={"knl": [0.0, 0.0, 2.0]}),
                                   el("u", "nllens", {"l": 0.0, "knll": 1.0, "cnll": 0.01})],
                      "updated": {"value0": True, "value1": True, "value2": True}, "tree": {"value0": ""}}}
    p = tmp_path / "ring.synergia.json"
    p.write_text(json.dumps(doc))
    assert guess_format(p) == "synergia"
    lat, rep = read(p)
    brho = lat.reference.brho_signed
    assert lat.reference.kinetic_energy_eV == pytest.approx(0.8e9, rel=1e-6)
    rb = lat.elements["rb"]
    assert rb.bend.rect and rb.bend.e1 == pytest.approx(0.05) and rb.bend.e2 == pytest.approx(0.04)
    q = lat.elements["q"]
    assert q.multipole.Bn[1] == pytest.approx(5.0 * brho) and q.multipole.tilt[1] == 0.1
    c = lat.elements["c"]
    assert c.rf.voltage_V == pytest.approx(1e6) and c.rf.phase_rad == pytest.approx(0.0) and c.rf.harmon == 84.0
    assert lat.elements["h"].hkick == 1e-3
    assert lat.elements["m"].multipole.BnL[2] == pytest.approx(2.0 * brho)
    assert lat.elements["u"].kind == "Marker" and "UNSUPPORTED_SYNERGIA_ELEMENT" in rep.codes()


@pytest.mark.skipif(not BOOSTER_JSON.is_file(), reason="Synergia clone with the Booster example not present")
def test_shipped_booster_archive_reads(tmp_path):
    lat, rep = read(BOOSTER_JSON, "synergia")
    kinds = [pl.element.kind for pl in lat.flatten()]
    assert kinds.count("Quadrupole") > 40 and kinds.count("RFCavity") >= 1
    assert lat.reference.species.name == "proton" and lat.reference.kinetic_energy_eV == pytest.approx(0.8e9, rel=1e-6)
    out = tmp_path / "booster.synergia.json"
    rep2 = write(lat, out, "synergia")
    assert rep2.ok
