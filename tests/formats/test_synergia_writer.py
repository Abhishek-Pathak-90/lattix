"""Synergia lattice JSON writer: rules coverage, the archive layout, the MAD-X-named attributes, the constant-p0
energy modes, goldens."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from lattix import read, write
from lattix.formats.base import check_rules_coverage
from lattix.formats.synergia import Writer
from lattix.formats.synergia.writer import TYPE_INDEX
from lattix.ir.elements import RFP, Drift, Foil, MagneticMultipoleP, Quadrupole, RFCavity
from lattix.ir.lattice import Lattice
from tests.formats.test_ocelot_writer import all_kinds_lattice, ref_at

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "synergia"


def _doc(lat: Lattice, tmp_path: Path, **opts):
    out = tmp_path / "lat.synergia.json"
    rep = write(lat, out, "synergia", **opts)
    return json.loads(out.read_text()), rep, out


def elements(doc: dict) -> dict[str, dict]:
    """``{name: {"stype": …, attribute: value, "lattix": tag}}`` of the archive's elements."""
    out = {}
    for e in doc["value0"]["elements"]:
        d = {"stype": e["stype"], "type": e["type"]}
        d.update({a["key"]: float(a["value"]["value0"]) for a in e["lazy_double_attributes"]})
        d.update({a["key"]: [float(x["value0"]) for x in a["value"]] for a in e["lazy_vector_attributes"]})
        d.update({a["key"]: a["value"] for a in e["string_attributes"]})
        out[e["name"]] = d
    return out


def _golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = path.read_text()
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(), f"golden {name} differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_archive_layout_and_reference(tmp_path):
    lat = all_kinds_lattice("proton")
    doc, rep, _ = _doc(lat, tmp_path)
    body = doc["value0"]
    assert body["name"] == "all_kinds" and body["has_reference_particle"] is True
    rp = body["reference_particle_value"]
    assert rp["charge"] == 1 and rp["four_momentum"]["mass"] == pytest.approx(0.93827208816)
    assert rp["four_momentum"]["energy"] == pytest.approx(0.94037208816)
    assert rp["four_momentum"]["gamma"] == pytest.approx(lat.reference.gamma)
    assert rp["state"] == {f"value{i}": 0.0 for i in range(6)} and "abs_time" in rp
    assert body["updated"] == {"value0": True, "value1": True, "value2": True} and body["tree"] == {"value0": ""}
    assert doc["lattix"]["energy_mode"] == "constant" and doc["lattix"]["rf_frequency_Hz"] == 162.5e6
    e0 = body["elements"][0]
    assert set(e0) == {"name", "format", "stype", "type", "ancestors", "string_attributes", "lazy_double_attributes",
                       "lazy_vector_attributes", "length_attribute_name", "bend_angle_attribute_name", "revision",
                       "markers"}
    assert e0["format"] == 1 and e0["markers"] == {f"value{i}": False for i in range(4)}
    for e in body["elements"]:
        assert e["type"] == TYPE_INDEX[e["stype"]]


def test_element_types_and_ledger(tmp_path):
    lat = all_kinds_lattice("proton")
    doc, rep, _ = _doc(lat, tmp_path)
    els = elements(doc)
    st = {n: d["stype"] for n, d in els.items()}
    assert st["q1"] == "quadrupole" and st["s1"] == "sextupole" and st["o1"] == "octupole"
    assert st["m1"] == "multipole" and st["b1"] == "sbend" and st["sol1"] == "solenoid"
    assert st["c1"] == "rfcavity" and st["c2"] == "rfcavity" and st["k1"] == "kicker"
    assert st["col1"] == "rcollimator" and st["mk1"] == "marker" and st["bpm1"] == "monitor"
    assert st["scr1"] == "instrument" and st["f1"] == "marker" and st["t1"] == "matrix"
    assert st["p1"] == "marker" and st["rc1"] == "marker" and st["fq1"] == "marker" and st["dir1"] == "marker"
    codes = rep.codes()
    for code in ("FOIL_TO_MARKER", "PATCH_DROPPED", "REFCHANGE_AS_TAG", "FOREIGN_DIRECTIVE", "CONST_P0",
                 "CONST_P0_START_RIGIDITY", "TAYLOR_BASIS_SYNERGIA", "APERTURE_AS_ATTRIBUTE", "TAYLOR_THIN_PLUS_DRIFT"):
        assert code in codes, code


def test_measured_conventions(tmp_path):
    from lattix.ir.walk import propagate

    lat = all_kinds_lattice("proton")
    doc, _, _ = _doc(lat, tmp_path)
    els = elements(doc)
    brho = lat.reference.brho_signed
    assert els["q1"]["k1"] == pytest.approx(1.2 / brho) and els["q1"]["hoffset"] == 1e-3
    assert els["s1"]["k2"] == pytest.approx(3.0 / brho) and els["o1"]["k3"] == pytest.approx(4.0 / brho)
    assert els["m1"]["knl"] == pytest.approx([0.01 / brho, 0.02 / brho])
    assert els["m1"]["ksl"] == pytest.approx([0.02 / brho, 0.0])
    b1 = els["b1"]
    assert b1["angle"] == 0.1 and b1["e1"] == 0.05 and b1["fint"] == 0.45 and b1["hgap"] == 0.03 and "k0" not in b1
    assert els["sol1"]["ks"] == pytest.approx(0.5 / brho)                           # MAD-X's ks = B/Bρ
    c1 = els["c1"]
    assert c1["volt"] == pytest.approx(0.08) and c1["freq"] == pytest.approx(162.5) and c1["l"] == 0.0
    assert c1["lag"] == pytest.approx((-85.0 / 360.0 + 0.25) % 1.0)                # V·sin(2π lag): lag = φ/2π + ¼
    assert "phase=" in c1["lattix"]
    k1 = next(p for p in propagate(lat) if p.element.name == "k1")
    r = k1.ref_in.brho_signed / brho                    # after the two gaps: the delta-mode kick scaling
    assert r > 1 and els["k1"]["hkick"] == pytest.approx(1e-3 * r) and els["k1"]["vkick"] == pytest.approx(-2e-3 * r)
    assert els["col1"]["xsize"] == 0.01 and els["col1"]["ysize"] == 0.02
    assert els["col1"]["aperture_type"] == "rectangular" and els["col1"]["rectangular_aperture_width"] == 0.02
    assert els["d4"]["aperture_type"] == "circular" and els["d4"]["circular_aperture_radius"] == 0.015
    # a thick Taylor: Synergia's matrix must be thin — the map with its drift divided out, then a drift body
    assert els["t1"]["l"] == 0.0 and els["t1_body"]["stype"] == "drift" and els["t1_body"]["l"] == 0.1
    assert "rm12" in els["t1"] and els["t1"]["rm11"] == 1.0 and "L=0.1" in els["t1"]["lattix"]
    assert els["rc1"]["lattix"].startswith("kind=ReferenceChange dE=1000")


def test_energy_modes_scale_the_downstream_strengths(tmp_path):
    """A 1 MV gap at 2.1 MeV then a quadrupole: Synergia scales every strength by p_design/p_bunch (MEASURED), so
    the default "constant" mode writes k1 = G/Bρ_start; "local" would double-correct; no phase slip on the lag."""
    from lattix.ir.walk import propagate

    lat = Lattice.from_sequence("g", [RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=0.0, frequency_Hz=162.5e6)),
                                      Drift(name="d", length=0.1),
                                      Quadrupole(name="q", length=0.1, multipole=MagneticMultipoleP(Bn={1: 1.0})),
                                      RFCavity(name="c2", rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30.0),
                                                                 frequency_Hz=162.5e6))],
                                ref_at(sp="proton"))
    q = next(p for p in propagate(lat) if p.element.name == "q")
    default, rep, _ = _doc(lat, tmp_path)
    local, _, _ = _doc(lat, tmp_path, energy_mode="local")
    assert elements(default)["q"]["k1"] == pytest.approx(1.0 / lat.reference.brho_signed)
    assert elements(local)["q"]["k1"] == pytest.approx(1.0 / q.ref_in.brho_signed)
    assert elements(default)["c"]["lag"] == pytest.approx(0.25)
    assert elements(default)["c2"]["lag"] == pytest.approx((-30.0 / 360.0 + 0.25) % 1.0)   # no slip
    assert "CONST_P0_START_RIGIDITY" in rep.codes() and "CONST_P0_PHASE_SLIP" not in rep.codes()


def test_bend_after_acceleration_is_recorded_as_underbent(tmp_path):
    from lattix.ir.elements import Bend, BendP

    lat = Lattice.from_sequence("gb", [RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=0.0, frequency_Hz=162.5e6)),
                                       Bend(name="b", length=1.0, bend=BendP(angle=0.1))], ref_at(sp="proton"))
    doc, rep, _ = _doc(lat, tmp_path)
    assert "k0" not in elements(doc)["b"] and "CONST_P0_BEND_UNDERBENT" in rep.codes()


def test_negative_species_flips_the_normalized_strengths(tmp_path):
    lat = all_kinds_lattice("h-")
    doc, _, _ = _doc(lat, tmp_path)
    els = elements(doc)
    assert lat.reference.brho_signed < 0 and els["q1"]["k1"] < 0 and els["sol1"]["ks"] < 0
    assert doc["value0"]["reference_particle_value"]["charge"] == -1


def test_strict_mode_raises_on_a_lossy_kind(tmp_path):
    from lattix.fidelity import TranslationError

    lat = Lattice.from_sequence("s", [Foil(name="f")], ref_at())
    with pytest.raises(TranslationError):
        write(lat, tmp_path / "f.synergia.json", "synergia", strict=True)


def test_goldens(tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.synergia.json"
    write(lat, out, "synergia")
    _golden("fodo.synergia.json", out)
    out2 = tmp_path / "all_kinds.synergia.json"
    write(all_kinds_lattice(), out2, "synergia")
    _golden("all_kinds.synergia.json", out2)


def test_vertical_bend_is_lossy(tmp_path):
    """libFF's sbend has no tilt (measured 2026-09-06): a tilted bend keeps the attribute for the round trip but
    is LOSSY, so the battery holds no engine pair to it."""
    from lattix.formats import write
    from lattix.ir.elements import Bend, BendP, Drift
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=8e8, rf_frequency_Hz=352.21e6)
    lat = Lattice.from_sequence("v", [Drift(name="d", length=0.5),
                                      Bend(name="bv", length=1.05, bend=BendP(angle=0.0416, e1=0.0208, e2=0.0208,
                                                                              tilt_ref=math.pi / 2)),
                                      Bend(name="bh", length=1.0, bend=BendP(angle=0.05))], ref)
    rep = write(lat, tmp_path / "v.synergia.json", "synergia")
    codes = {(e.element, e.code) for e in rep.entries if e.cls.value == "LOSSY"}
    assert ("bv", "BEND_TILT_DROPPED") in codes and not any(el == "bh" for el, _ in codes)
