"""Cheetah LatticeJSON reader: round trips, fixed points, the reference, folding, conventions."""
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
from lattix.ir.reference import ReferenceParticle, species
from tests.formats.test_cheetah_writer import all_kinds_lattice, ref_at

DATA = Path(__file__).resolve().parents[1] / "data" / "public"


def _cycle(lat: Lattice, tmp_path: Path, **read_opts):
    out = tmp_path / "a.cheetah.json"
    rep = write(lat, out, "cheetah")
    back, rep2 = read(out, **read_opts)
    return out, rep, back, rep2


@pytest.mark.parametrize("sp", ["proton", "h-"])
def test_all_kinds_round_trip_and_fixed_point(tmp_path, sp):
    lat = all_kinds_lattice(sp)
    out, rep, back, rep2 = _cycle(lat, tmp_path)
    assert back.reference.species.name.lower() == sp and back.reference.kinetic_energy_eV == 2.1e6
    diff = compare_profiles(profile(lat, neutral_set(rep)), profile(back, neutral_set(rep)))
    assert diff.ok, diff.problems[:5]
    kinds = {e.kind for e in back.elements.values()}
    assert {"Quadrupole", "Sextupole", "Bend", "Solenoid", "RFCavity", "Kicker", "Collimator", "Instrument",
            "Foil", "Taylor", "Patch", "ReferenceChange", "Freq", "Directive", "Multipole", "Marker"} <= kinds
    out2 = tmp_path / "b.cheetah.json"
    write(back, out2, "cheetah")
    a = json.loads(out.read_text())
    b = json.loads(out2.read_text())
    a.pop("lattix")
    b.pop("lattix")
    for d in (a, b):
        for entry in d["elements"].values():
            entry[1]["metadata"]["lattix"].pop("original_type", None)
    assert a == b


def test_apertures_fold_back_and_the_collimator_body_returns(tmp_path):
    lat = all_kinds_lattice()
    _, _, back, _ = _cycle(lat, tmp_path)
    d2 = back.elements["d2"]
    assert d2.aperture is not None and d2.aperture.aperture_at == "BOTH_ENDS"
    assert d2.aperture.x_limits == pytest.approx((-0.015, 0.015))
    col = back.elements["col1"]
    assert col.kind == "Collimator" and col.length == pytest.approx(0.05)
    assert "col1_body" not in back.elements and "d2_aper_in" not in back.elements


def test_reference_from_tag_or_options(tmp_path):
    lat = Lattice.from_sequence("s", [Drift(name="d", length=1.0)], ref_at())
    out, _, back, rep2 = _cycle(lat, tmp_path)
    assert "REFERENCE_FROM_TAG" in rep2.codes()
    doc = json.loads(out.read_text())
    doc["info"] = "no tag here"
    (tmp_path / "notag.cheetah.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="species"):
        read(tmp_path / "notag.cheetah.json")
    lat2, rep3 = read(tmp_path / "notag.cheetah.json", species="electron", kinetic_energy_eV=5e6)
    assert lat2.reference.species.charge == -1 and "REFERENCE_FROM_TAG" not in rep3.codes()


def test_rbend_faces_and_cheetah_conventions(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=8e8)
    brho = ref.brho_signed
    doc = {"version": "cheetah-0.8", "title": "t", "info": "", "root": "r",
           "elements": {"rb": ["RBend", {"length": 0.8, "angle": 0.08, "rbend_e1": 0.01, "rbend_e2": 0.02}],
                        "q": ["Quadrupole", {"length": 0.2, "k1": 5.0, "tilt": 0.1, "misalignment": [1e-3, 0.0]}],
                        "s": ["Solenoid", {"length": 0.5, "k": 1.0}],
                        "c": ["Cavity", {"length": 0.02, "voltage": -1e6, "phase": 30.0, "frequency": 1.625e8}],
                        "h": ["HorizontalCorrector", {"length": 0.1, "angle": 1e-3}],
                        "u": ["Undulator", {"length": 0.5}]},
           "lattices": {"r": ["rb", "q", "s", "c", "h", "u"]}}
    p = tmp_path / "hand.cheetah.json"
    p.write_text(json.dumps(doc))
    lat, rep = read(p, species="proton", kinetic_energy_eV=8e8)
    rb = lat.elements["rb"]
    assert rb.bend.rect and rb.bend.e1 == pytest.approx(0.05) and rb.bend.e2 == pytest.approx(0.06)
    q = lat.elements["q"]
    assert q.multipole.Bn[1] == pytest.approx(5.0 * brho) and q.multipole.tilt[1] == 0.1 and q.shift.x_offset == 1e-3
    assert lat.elements["s"].solenoid.Bsol_T == pytest.approx(2.0 * brho)
    c = lat.elements["c"]
    assert c.rf.voltage_V == pytest.approx(1e6) and c.rf.phase_rad == pytest.approx(math.radians(-30.0))
    assert lat.elements["h"].kind == "Kicker" and lat.elements["h"].hkick == 1e-3
    assert lat.elements["u"].kind == "Drift" and "UNSUPPORTED_CHEETAH_ELEMENT" in rep.codes()


def test_nested_lattices_become_lines(tmp_path):
    doc = {"version": "cheetah-0.8", "title": "t", "info": "", "root": "ring",
           "elements": {"d": ["Drift", {"length": 1.0}], "q": ["Quadrupole", {"length": 0.2, "k1": 1.0}]},
           "lattices": {"cell": ["d", "q"], "ring": ["cell", "cell", "d"]}}
    p = tmp_path / "nested.cheetah.json"
    p.write_text(json.dumps(doc))
    lat, _ = read(p, species="proton", kinetic_energy_eV=1e9)
    assert lat.use == "ring" and [it.ref for it in lat.lines["ring"].items] == ["cell", "cell", "d"]
    assert sum(pl.element.length for pl in lat.flatten()) == pytest.approx(3.4)


def test_bare_json_is_sniffed(tmp_path):
    lat = Lattice.from_sequence("s", [Drift(name="d", length=1.0)], ref_at())
    out = tmp_path / "plain.json"
    write(lat, out, "cheetah")
    assert guess_format(out) == "cheetah"
