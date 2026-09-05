"""SciBmad (Beamlines.jl) writer: rules coverage, the Julia text, goldens, tags, energy modes."""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from lattix import read, write
from lattix.formats.base import check_rules_coverage
from lattix.formats.scibmad import GAIN_SIGN, Reader, Writer
from lattix.formats.scibmad.writer import RESERVED, NameMap, sanitize, species_expr
from lattix.ir.elements import (
    RFP,
    Bend,
    BendP,
    Drift,
    Kicker,
    MagneticMultipoleP,
    Quadrupole,
    ReferenceChange,
    RFCavity,
    Solenoid,
    SolenoidP,
    Taylor,
)
from lattix.ir.energy_mode import phase_slip_turns
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.walk import propagate
from tests.formats.test_impactx_writer import all_kinds_lattice

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "scibmad"


def _norm(text: str) -> str:
    return re.sub(r"lattix [0-9][0-9a-z.+-]*", "lattix <version>", text)


def assert_golden(name: str, text: str) -> None:
    import os

    path = GOLDEN / name
    if os.environ.get("LATTIX_UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_norm(text))
    assert path.exists(), f"missing golden {path}; rerun with LATTIX_UPDATE_GOLDEN=1"
    assert _norm(text) == path.read_text()


def proton(ke: float = 2.1e6) -> ReferenceParticle:
    return ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke, rf_frequency_Hz=162.5e6)


# ------------------------------------------------------------------ rules / names
def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_sanitize_makes_julia_identifiers():
    assert sanitize("QF") == "QF"                       # case kept: Julia is case sensitive
    assert sanitize("BR.QFO11") == "BR_QFO11"
    assert sanitize("1abc") == "e_1abc"
    assert sanitize("end") == "end_x" and sanitize("Drift") == "Drift_x"
    assert sanitize("") == "e_"
    names = NameMap()
    assert names.assign("q.1") == "q_1" and names.assign("q_1") == "q_1_2"
    assert names.renamed == {"q_1": "q.1", "q_1_2": "q_1"}
    for word in RESERVED:
        assert sanitize(word) != word


def test_species_expressions():
    assert species_expr(species("proton")) == ('Species("proton")', True)
    assert species_expr(species("h-")) == ('Species("#1H-")', True)       # m_p + 2 m_e, charge -1
    text, known = species_expr(species("proton").model_copy(update={"name": "ion_A238_Q33", "charge": 33,
                                                                     "mass_eV": 2.2e11}))
    assert not known and text.startswith('Species("ion_A238_Q33", 33, 220000000000')


# ------------------------------------------------------------------ the text
def test_fodo_text(tmp_path):
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.jl"
    rep = write(lat, out, "scibmad")
    assert rep.ok and rep.codes() == {}
    text = out.read_text()
    assert "using Beamlines" in text
    assert "@elements begin" in text and text.count("\nend\n") == 1
    assert "qf = Quadrupole(L = 0.3, Kn1 = 0.6)" in text
    # the design bend needs its dipole field: g_ref alone is a curved frame (measured)
    assert "b1 = SBend(L = 1, g_ref = 0.1, Kn0 = 0.1, e1 = 0.05, e2 = 0.05)" in text
    assert re.search(r"fodo = Beamline\(\[qf, drift_0, b1, .*\];\n    pc_ref = 1463295949\.06973, "
                     r'species_ref = Species\("proton"\)\)', text)
    assert "# lattix: energy_mode=delta" in text
    assert_golden("fodo.jl", text)


def test_cavity_voltage_carries_the_measured_gain_sign(tmp_path):
    lat = Lattice.from_sequence("c", [RFCavity(name="cav", length=0.0,
                                               rf=RFP(voltage_V=3e5, phase_rad=-math.pi / 6, frequency_Hz=162.5e6))],
                                proton())
    text = Writer().dumps(lat)
    assert GAIN_SIGN == -1.0
    assert f"cav = RFCavity(L = 0, voltage = {-3e5:.15g}, phi0 = {-math.pi / 6:.15g}, rf_frequency = 162500000)" in text


def test_all_kinds_writes_every_kind_and_the_ledger_is_complete(tmp_path):
    lat = all_kinds_lattice()
    out = tmp_path / "all.jl"
    rep = write(lat, out, "scibmad")
    text = out.read_text()
    names = {p.element.name for p in lat.flatten()}
    seen = {e.element for e in rep.entries}
    assert names <= seen, names - seen
    for kind in ("Drift(", "Quadrupole(", "Sextupole(", "Octupole(", "Multipole(", "SBend(", "Solenoid(",
                 "RFCavity(", "Kicker(", "Patch(", "LineElement(", "Marker()"):
        assert kind in text, kind
    assert "function lattix_map_" in text and "transport_map = lattix_map_" in text
    assert 'kind="ReferenceChange"' in text and 'kind="Foil"' in text and 'kind="Instrument"' in text
    assert "aperture_shape = ApertureShape." in text
    assert_golden("all_kinds.jl", text)


def test_downgrades_are_never_silent():
    lat = all_kinds_lattice()
    rep = Writer().write(lat, Path("/dev/null"))
    codes = rep.codes()
    for code in ("FOIL_TO_MARKER", "REFCHANGE_AS_TAG", "NCELLS_TO_DRIFT", "RFQ_TO_DRIFT",
                 "SUPERPOSITION_FLATTENED", "FOREIGN_DIRECTIVE"):
        assert code in codes, code
    assert not rep.ok


def test_fringe_integrals_are_the_product_with_hgap_in_the_tag():
    b = Bend(name="b", length=1.0, bend=BendP(angle=0.1, e1=0.02, e2=0.03, edge_int1=0.45, edge_int2=0.5,
                                              hgap=0.03, rect=True))
    text = Writer().dumps(Lattice.from_sequence("b", [b], proton(8e8)))
    assert "edge1_int = 0.0135, edge2_int = 0.015" in text
    assert '# lattix: name="b" hgap="0.03" rect="true"' in text


def test_kicker_and_solenoid_conventions():
    ref = proton(8e8)
    k = Kicker(name="k", hkick=1e-3, vkick=-2e-3)
    s = Solenoid(name="s", length=0.5, solenoid=SolenoidP(Bsol_T=0.4))
    text = Writer().dumps(Lattice.from_sequence("ks", [k, s], ref))
    assert "k = Kicker(Kn0L = -0.001, Ks0L = -0.002)" in text          # measured: Kn0L = -hkick, Ks0L = +vkick
    assert f"s = Solenoid(L = 0.5, Ksol = {0.4 / ref.brho_signed:.15g})" in text


def test_taylor_map_function_is_linear_julia():
    t = Taylor(name="tm")
    t.matrix[1][0] = 0.7
    t.matrix[0][1] = 0.02
    t.offset[1] = 1e-3
    text = Writer().dumps(Lattice.from_sequence("t", [Drift(name="d", length=0.1), t], proton(8e8)))
    assert "function lattix_map_tm(v, q, p=nothing)" in text
    assert "    v1 = 1*v[1] + 0.02*v[2]" in text
    assert "    v2 = 0.7*v[1] + 1*v[2] + 0.001" in text
    assert "    return (v1, v2, v3, v4, v5, v6), q" in text
    assert "tm = LineElement(L = 0, transport_map = lattix_map_tm)" in text


def test_delta_mode_scales_strengths_after_rf_and_moves_downstream_phases(tmp_path):
    ref = proton()
    brho = ref.brho_signed
    els = [RFCavity(name="g1", length=0.0, rf=RFP(voltage_V=1e6, phase_rad=0.0, frequency_Hz=162.5e6)),
           Drift(name="d1", length=0.3),
           Quadrupole(name="q1", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho})),
           RFCavity(name="g2", length=0.0, rf=RFP(voltage_V=1e6, phase_rad=-math.pi / 6, frequency_Hz=162.5e6)),
           ReferenceChange(name="jump", dE_ref_eV=0.5e6),
           Quadrupole(name="q2", length=0.2, multipole=MagneticMultipoleP(Bn={1: 5.0 * brho}))]
    lat = Lattice.from_sequence("lin", els, ref)
    out = tmp_path / "lin.jl"
    rep = write(lat, out, "scibmad")
    codes = rep.codes()
    # three accelerating items: the two gaps and the reference jump (as the MAD-X writer counts them)
    assert codes["CONST_P0_DELTA_RIGIDITY"] == 3 and codes["CONST_P0_PHASE_SLIP"] == 1
    # only g2 is off crest: the thin-gap defocusing is ∝ sin φ, so g1 (crest) gets no lens
    assert codes["REFCHANGE_AS_TAG"] == 1 and codes["THIN_GAP_RF_FOCUSING_AS_MATRIX"] == 1
    text = out.read_text()
    # q1 sits after a pure RF gain: k1 = G / Bρ_start (the engine's particle carries the rest as pz)
    assert re.search(r"q1 = Quadrupole\(L = 0\.2, Kn1 = 5\)", text)
    placed = {p.element.name: p for p in propagate(lat)}
    slip = phase_slip_turns(placed["g2"].ref_in, placed["g2"].s_in, 0.0, 162.5e6, ref)
    assert f"phi0 = {-math.pi / 6 - 2 * math.pi * slip:.15g}" in text
    assert "g2_rfdefocus = LineElement(L = 0, transport_map = lattix_map_g2_rfdefocus)" in text
    back, rep2 = Reader().read(out)
    assert "PHASE_SLIP_RESTORED" in rep2.codes() and "ENERGY_MODE_RESTORED" in rep2.codes()
    for q in ("q1", "q2"):
        assert back.elements[q].multipole.Bn[1] == pytest.approx(5.0 * brho, rel=1e-11)
    assert back.elements["g2"].rf.phase_rad == pytest.approx(-math.pi / 6, abs=1e-12)
    assert back.elements["jump"].kind == "ReferenceChange" and back.elements["jump"].dE_ref_eV == 0.5e6


def test_invalid_energy_mode_is_refused():
    with pytest.raises(ValueError, match="energy_mode"):
        Writer().write(all_kinds_lattice(), Path("/dev/null"), energy_mode="nope")


def test_strict_mode_raises_on_the_first_loss():
    from lattix.fidelity import TranslationError

    with pytest.raises(TranslationError):
        Writer().write(all_kinds_lattice(), Path("/dev/null"), strict=True)
