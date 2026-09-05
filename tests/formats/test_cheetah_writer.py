"""Cheetah LatticeJSON writer: rules coverage, the document, the measured conventions, goldens."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from lattix import read, write
from lattix.formats.base import check_rules_coverage
from lattix.formats.cheetah import Writer
from lattix.formats.cheetah.writer import NameMap
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    Foil,
    Freq,
    Instrument,
    Kicker,
    MagneticMultipoleP,
    Marker,
    Multipole,
    Octupole,
    Patch,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    Sextupole,
    Solenoid,
    SolenoidP,
    Taylor,
)
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "cheetah"
KE = 2.1e6


def ref_at(ke=KE, sp="proton", f=162.5e6):
    return ReferenceParticle(species=species(sp), kinetic_energy_eV=ke, rf_frequency_Hz=f)


def all_kinds_lattice(sp="proton") -> Lattice:
    ref = ref_at(sp=sp)
    els = [
        Drift(name="d1", length=0.5),
        Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 1.2}), shift=BodyShiftP(x_offset=1e-3)),
        Sextupole(name="s1", length=0.1, multipole=MagneticMultipoleP(Bn={2: 3.0})),
        Octupole(name="o1", length=0.1, multipole=MagneticMultipoleP(Bn={3: 4.0})),
        Multipole(name="m1", multipole=MagneticMultipoleP(BnL={0: 0.01}, BsL={0: 0.02})),
        Bend(name="b1", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.45, hgap=0.03)),
        Solenoid(name="sol1", length=0.3, solenoid=SolenoidP(Bsol_T=0.5)),
        RFCavity(name="c1", length=0.0, rf=RFP(voltage_V=8e4, phase_rad=math.radians(-85.0), frequency_Hz=162.5e6)),
        RFCavity(name="c2", length=0.2, rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30.0), frequency_Hz=162.5e6,
                                               cavity_type="TRAVELING_WAVE")),
        Kicker(name="k1", length=0.1, hkick=1e-3, vkick=-2e-3),
        Collimator(name="col1", length=0.05, aperture=ApertureP.rect(0.01, 0.02)),
        Marker(name="mk1"),
        Instrument(name="bpm1", family="BPM"),
        Instrument(name="scr1", family="SCREEN"),
        Foil(name="f1", material="C", thickness_kg_per_m2=1e-3),
        Taylor(name="t1", length=0.1, matrix=[[1.0 if i == j else (0.1 if (i, j) == (0, 1) else 0.0) for j in range(6)]
                                              for i in range(6)]),
        Patch(name="p1", x_offset=1e-3),
        ReferenceChange(name="rc1", dE_ref_eV=1e3),
        Freq(name="fq1", frequency_Hz=162.5e6),
        Directive(name="dir1", format="tracewin", card="LATTICE", args=["4", "0"], role="period_start"),
        Drift(name="d2", length=0.2, aperture=ApertureP.circle(0.015)),
    ]
    return Lattice.from_sequence("all_kinds", els, ref)


def _doc(lat: Lattice, tmp_path: Path, **opts):
    out = tmp_path / "lat.cheetah.json"
    rep = write(lat, out, "cheetah", **opts)
    return json.loads(out.read_text()), rep, out


def _golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = path.read_text()
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(), f"golden {name} differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_all_kinds_document_types_and_ledger(tmp_path):
    lat = all_kinds_lattice()
    doc, rep, out = _doc(lat, tmp_path)
    assert doc["version"] == "cheetah-0.8" and doc["root"] == "all_kinds"
    types = {name: entry[0] for name, entry in doc["elements"].items()}
    assert types["q1"] == "Quadrupole" and types["s1"] == "Sextupole" and types["o1"] == "Drift"
    assert types["m1"] == "CombinedCorrector" and types["b1"] == "Dipole" and types["sol1"] == "Solenoid"
    assert types["c1"] == "Cavity" and types["k1"] == "CombinedCorrector" and types["col1"] == "Aperture"
    assert types["col1_body"] == "Drift" and types["bpm1"] == "BPM" and types["scr1"] == "Marker"
    assert types["t1"] == "CustomTransferMap" and types["fq1"] == "Marker" and types["d2_aper_in"] == "Aperture"
    order = doc["lattices"]["all_kinds"]
    assert order.index("d2_aper_in") < order.index("d2") < order.index("d2_aper_out")
    codes = rep.codes()
    for code in ("OCTUPOLE_TO_DRIFT", "MULTIPOLE_AS_CORRECTOR", "CHEETAH_ZERO_LENGTH_CAVITY", "FOIL_TO_MARKER",
                 "PATCH_DROPPED", "REFCHANGE_DROPPED", "FOREIGN_DIRECTIVE", "APERTURE_AS_ELEMENT",
                 "INSTRUMENT_AS_MARKER", "TAYLOR_BASIS_CHEETAH"):
        assert code in codes, code
    assert "# lattix: reference" in doc["info"]
    assert doc["lattix"]["source_format"] == "IR"


def test_measured_conventions(tmp_path):
    lat = all_kinds_lattice()
    doc, _, _ = _doc(lat, tmp_path)
    ref = lat.reference
    brho = ref.brho_signed
    q1 = doc["elements"]["q1"][1]
    assert q1["k1"] == pytest.approx(1.2 / brho) and q1["misalignment"] == [1e-3, 0.0]
    assert doc["elements"]["sol1"][1]["k"] == pytest.approx(0.5 / (2 * brho))           # B0/(2Bρ)
    b1 = doc["elements"]["b1"][1]
    assert b1["gap"] == pytest.approx(0.06) and b1["fringe_integral"] == 0.45 and b1["dipole_e1"] == 0.05
    c1 = doc["elements"]["c1"][1]
    assert c1["voltage"] == pytest.approx(-8e4) and c1["phase"] == pytest.approx(+85.0)   # proton: −V, phase = −φ
    assert doc["elements"]["c2"][1]["cavity_type"] == "traveling_wave"
    k1 = doc["elements"]["k1"][1]
    assert k1["horizontal_angle"] == 1e-3 and k1["vertical_angle"] == -2e-3
    m1 = doc["elements"]["m1"][1]
    assert m1["horizontal_angle"] == pytest.approx(-0.01 / brho) and m1["vertical_angle"] == pytest.approx(0.02 / brho)
    col = doc["elements"]["col1"][1]
    assert col["x_max"] == 0.01 and col["y_max"] == 0.02 and col["shape"] == "rectangular"
    tm = doc["elements"]["t1"][1]["predefined_transfer_map"]
    assert len(tm) == 7 and tm[6] == [0.0] * 6 + [1.0] and tm[0][1] == pytest.approx(0.1)


def test_negative_species_flips_the_signed_quantities(tmp_path):
    lat = all_kinds_lattice("h-")
    doc, _, _ = _doc(lat, tmp_path)
    brho = lat.reference.brho_signed
    assert brho < 0
    assert doc["elements"]["q1"][1]["k1"] == pytest.approx(1.2 / brho)
    assert doc["elements"]["q1"][1]["k1"] < 0                                   # same field, opposite focusing
    assert doc["elements"]["c1"][1]["voltage"] == pytest.approx(+8e4)          # −V/q with q = −1
    assert doc["elements"]["sol1"][1]["k"] < 0                                  # rotation follows the charge


def test_names_are_python_identifiers_and_reversible(tmp_path):
    nm = NameMap()
    assert nm.assign("bi4.bsw1l1.1") == "bi4_bsw1l1_1" and nm.assign("1q") == "e_1q"
    assert nm.assign("bi4.bsw1l1.1") == "bi4_bsw1l1_1_2"
    ref = ref_at()
    lat = Lattice.from_sequence("s", [Drift(name="a b", length=0.1)], ref)
    doc, rep, _ = _doc(lat, tmp_path)
    assert "a_b" in doc["elements"] and doc["elements"]["a_b"][1]["metadata"]["lattix"]["name"] == "a b"
    assert "NAMES_SANITIZED" in [e.code for e in rep.entries]


def test_field_map_becomes_the_replacement_cavity(tmp_path):
    from lattix.ir.elements import FieldMap

    ref = ref_at()
    fm = FieldMap(name="fm", length=0.4, geom=100, files=["x"], rf=RFP(frequency_Hz=162.5e6, voltage_V=5e5,
                                                                        phase_rad=0.3, dE_ref_eV=3e5))
    fm.meta["map_summary"] = {"kind": "rf", "v_c_V": 5.1e5, "phase_sync_rad": -0.4, "dE_ref_eV": 3e5}
    lat = Lattice.from_sequence("s", [Drift(name="d", length=0.1), fm], ref)
    doc, rep, _ = _doc(lat, tmp_path)
    assert doc["elements"]["fm"][0] == "Cavity" and doc["elements"]["fm"][1]["voltage"] == pytest.approx(-5.1e5)
    assert doc["elements"]["fm"][1]["phase"] == pytest.approx(-math.degrees(-0.4))     # phase = −φs
    assert "FM_TO_CAVITY" in rep.codes()


def test_strict_mode_raises_on_a_lossy_kind(tmp_path):
    from lattix.fidelity import TranslationError

    lat = Lattice.from_sequence("s", [Octupole(name="o", length=0.1)], ref_at())
    with pytest.raises(TranslationError):
        write(lat, tmp_path / "o.cheetah.json", "cheetah", strict=True)


def test_goldens(tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.cheetah.json"
    write(lat, out, "cheetah")
    _golden("fodo.cheetah.json", out)
    out2 = tmp_path / "all_kinds.cheetah.json"
    write(all_kinds_lattice(), out2, "cheetah")
    _golden("all_kinds.cheetah.json", out2)
