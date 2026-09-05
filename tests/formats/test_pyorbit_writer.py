"""PyORBIT3 linac XML writer: rules coverage, the document, the measured conventions, goldens."""
from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from lattix import read, write
from lattix.fidelity import TranslationError
from lattix.formats.base import check_rules_coverage
from lattix.formats.pyorbit import Writer
from lattix.ir.elements import (
    RFP,
    ApertureP,
    Bend,
    BendP,
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
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "pyorbit"
KE = 2.1e6


def ref_at(ke=KE, sp="proton", f=162.5e6):
    return ReferenceParticle(species=species(sp), kinetic_energy_eV=ke, rf_frequency_Hz=f)


def all_kinds_lattice(sp="proton") -> Lattice:
    ref = ref_at(sp=sp)
    els = [
        Drift(name="d1", length=0.5),
        Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 1.2}), aperture=ApertureP.circle(0.015)),
        Sextupole(name="s1", length=0.1, multipole=MagneticMultipoleP(Bn={2: 3.0})),
        Octupole(name="o1", length=0.1, multipole=MagneticMultipoleP(Bn={3: 4.0})),
        Multipole(name="m1", multipole=MagneticMultipoleP(BnL={0: 0.01}, BsL={0: 0.02})),
        Bend(name="b1", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05)),
        Solenoid(name="sol1", length=0.3, solenoid=SolenoidP(Bsol_T=0.5)),
        RFCavity(name="c1", length=0.0, rf=RFP(voltage_V=8e4, phase_rad=math.radians(-85.0), frequency_Hz=162.5e6)),
        RFCavity(name="c2", length=0.2, rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30.0), frequency_Hz=325e6)),
        Kicker(name="k1", length=0.1, hkick=1e-3, vkick=-2e-3),
        Collimator(name="col1", length=0.05, aperture=ApertureP.rect(0.01, 0.02)),
        Marker(name="mk1"),
        Instrument(name="bpm1", family="BPM"),
        Foil(name="f1", material="C", thickness_kg_per_m2=1e-3),
        Taylor(name="t1", length=0.1),
        Patch(name="p1", x_offset=1e-3),
        ReferenceChange(name="rc1", dE_ref_eV=1e3),
        Freq(name="fq1", frequency_Hz=162.5e6),
        Directive(name="dir1", format="tracewin", card="LATTICE", args=["4", "0"], role="period_start"),
        Drift(name="d2", length=0.2),
    ]
    return Lattice.from_sequence("all_kinds", els, ref)


def _doc(lat: Lattice, tmp_path: Path, **opts):
    out = tmp_path / "lat.pyorbit.xml"
    rep = write(lat, out, "pyorbit", **opts)
    root = ET.fromstring(out.read_text())
    seq = next(c for c in root if c.get("length") is not None)
    elements = {e.get("name"): e for e in seq.findall("accElement")}
    return root, seq, elements, rep, out


def _golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = path.read_text()
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(), f"golden {name} differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_document_structure_and_ledger(tmp_path):
    lat = all_kinds_lattice()
    root, seq, els, rep, out = _doc(lat, tmp_path)
    assert seq.tag == "all_kinds"
    assert float(seq.get("length")) == pytest.approx(sum(p.element.length for p in lat.flatten()))
    types = {n: e.get("type") for n, e in els.items()}
    assert types["q1"] == "QUAD" and types["b1"] == "BEND" and types["sol1"] == "SOLENOID"
    assert types["c1"] == "RFGAP" and types["c2"] == "RFGAP" and types["k1"] == "DCH" and types["k1_V"] == "DCV"
    assert types["m1"] == "DCH" and types["mk1"] == "MARKER" and types["bpm1"] == "MARKER" and types["col1"] == "MARKER"
    assert "d1" not in els and "d2" not in els and "s1" not in els and "p1" not in els      # implicit / dropped
    cavs = seq.find("Cavities").findall("Cavity")
    assert len(cavs) == 2 and {float(c.get("frequency")) for c in cavs} == {162.5e6, 325e6}   # one cavity per gap
    assert "# lattix: reference" in out.read_text()
    codes = rep.codes()
    for code in ("PYORBIT_NO_MULTIPOLE", "PYORBIT_MULTIPOLE_AS_CORRECTOR", "THICK_CAVITY_AS_GAP",
                 "THICK_KICKER_SPLIT", "COLLIMATOR_TO_MARKER",
                 "INSTRUMENT_AS_MARKER", "FOIL_TO_MARKER", "TAYLOR_DROPPED", "PATCH_DROPPED", "REFCHANGE_DROPPED",
                 "FOREIGN_DIRECTIVE"):
        assert code in codes, code


def test_measured_conventions(tmp_path):
    lat = all_kinds_lattice()
    _, _, els, _, _ = _doc(lat, tmp_path)
    from lattix.ir.walk import propagate

    brho = lat.reference.brho_signed
    brho_k1 = next(p.ref_in.brho_signed for p in propagate(lat) if p.element.name == "k1")   # after the cavities
    q = els["q1"].find("parameters")
    assert float(q.get("field")) == 1.2 and q.get("aprt_type") == "1"
    assert float(q.get("aperture")) == pytest.approx(0.03)
    assert float(els["q1"].get("pos")) == pytest.approx(0.6)                 # centre: 0.5 m drift + 0.1
    b = els["b1"].find("parameters")
    assert float(b.get("theta")) == 0.1 and float(b.get("ea1")) == 0.05 and float(b.get("ea2")) == 0.05
    s = els["sol1"].find("parameters")
    assert float(s.get("B")) == pytest.approx(0.5 / abs(brho))                 # B₀/Bρ in 1/m
    c1 = els["c1"].find("parameters")
    assert float(c1.get("E0TL")) == pytest.approx(8e4 * 1e-9) and float(c1.get("phase")) == pytest.approx(-85.0)
    assert float(els["c2"].get("pos")) == pytest.approx(float(els["c2"].get("pos")))
    k = els["k1"].find("parameters")
    kv = els["k1_V"].find("parameters")
    eff = float(k.get("effLength"))
    assert eff == pytest.approx(0.1)
    assert float(k.get("B")) == pytest.approx(-1e-3 * brho_k1 / eff)         # x' -= B·L/Bρ (local rigidity)
    assert float(kv.get("B")) == pytest.approx(-2e-3 * brho_k1 / eff)        # y' += B·L/Bρ
    m = els["m1"].find("parameters")
    assert float(m.get("effLength")) == pytest.approx(1e-3)                    # a thin kicker: 1 mm effLength


def test_negative_species_flips_the_gap_phase(tmp_path):
    lat = all_kinds_lattice("h-")
    _, _, els, _, _ = _doc(lat, tmp_path)
    assert float(els["c1"].find("parameters").get("phase")) == pytest.approx(95.0)      # −85° + 180°
    assert float(els["sol1"].find("parameters").get("B")) > 0          # B₀/|Bρ|: the charge is PyORBIT's job


def test_touching_magnets_never_overlap_in_pyorbits_arithmetic(tmp_path):
    ref = ref_at()
    els = [Quadrupole(name=f"q{i}", length=0.0305000000000001, multipole=MagneticMultipoleP(Bn={1: 1.0}))
           for i in range(4)]
    lat = Lattice.from_sequence("touch", els, ref)
    _, _, elements, _, _ = _doc(lat, tmp_path)
    prev = 0.0
    for name in ("q0", "q1", "q2", "q3"):
        pos, L = float(elements[name].get("pos")), float(elements[name].get("length"))
        assert pos - L / 2.0 >= prev                              # what the factory computes
        prev = pos + L / 2.0


def test_strict_mode_raises_on_a_lossy_kind(tmp_path):
    lat = Lattice.from_sequence("s", [Sextupole(name="s", length=0.1)], ref_at())
    with pytest.raises(TranslationError):
        write(lat, tmp_path / "s.pyorbit.xml", "pyorbit", strict=True)


def test_goldens(tmp_path):
    lat, _ = read(DATA / "helix" / "mebt_line.dat", "tracewin", species="h-", kinetic_energy_eV=2.1e6,
                  frequency_Hz=162.5e6)
    out = tmp_path / "mebt_line.pyorbit.xml"
    write(lat, out, "pyorbit")
    _golden("mebt_line.pyorbit.xml", out)
    out2 = tmp_path / "all_kinds.pyorbit.xml"
    write(all_kinds_lattice(), out2, "pyorbit")
    _golden("all_kinds.pyorbit.xml", out2)
    assert not re.search(r'length="-', out2.read_text())
