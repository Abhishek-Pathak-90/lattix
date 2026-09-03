"""Engine-free tests of the IR foundation."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lattix.fidelity import FidelityClass, FidelityReport, TranslationError
from lattix.ir import (
    ALL_KINDS,
    RFP,
    Bend,
    BendP,
    Drift,
    Freq,
    Lattice,
    Line,
    LineItem,
    Quadrupole,
    ReferenceChange,
    ReferenceParticle,
    RFCavity,
    evaluate,
    evaluate_rpn,
    propagate,
    species,
    survey,
)
from lattix.ir import rf as rfconv
from lattix.ir.expr import ExpressionError, LazyResolver
from lattix.ir.normalize import gradient_from_k1, k1_from_gradient

PROTON = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)


def test_reference_particle_kinematics():
    r = ReferenceParticle(species=species("h-"), kinetic_energy_eV=800e6)
    assert r.brho_signed == pytest.approx(-4.8829, abs=2e-4)     # PIP-II BTL header: mad2tw -4.8828922
    assert r.brho_abs == -r.brho_signed
    assert ReferenceParticle.from_brho("h-", 4.881).kinetic_energy_eV == pytest.approx(799.5e6, rel=2e-3)
    assert ReferenceParticle.from_total_energy("proton", 1.738272e9).kinetic_energy_eV == pytest.approx(800e6, rel=1e-6)
    assert r.advanced(dE_eV=1e6).kinetic_energy_eV == 801e6
    assert species("H_MINUS").name == "h-"


def test_expressions():
    assert evaluate("2*pi^2 + sqrt(16)", {}) == pytest.approx(2 * math.pi**2 + 4)
    assert evaluate("kq*lq", {"kq": 2.0, "lq": 0.3}) == pytest.approx(0.6)
    assert evaluate_rpn("2 3 * pi /") == pytest.approx(6 / math.pi)
    assert evaluate_rpn("30 dsin") == pytest.approx(0.5)
    with pytest.raises(ExpressionError):
        evaluate("__import__('os')", {})
    lz = LazyResolver({"a": "2*b", "b": "c+1", "c": "3"})
    assert lz("A") == 8.0
    with pytest.raises(ExpressionError):
        LazyResolver({"x": "y", "y": "x"})("x")


def test_flatten_repeat_and_reverse():
    d = Drift(name="d", length=1.0)
    b = Bend(name="b", length=1.0, bend=BendP(angle=0.1, e1=0.02, e2=0.05))
    lat = Lattice(reference=PROTON, elements={"d": d, "b": b},
                  lines={"cell": Line(name="cell", items=[LineItem(ref="d"), LineItem(ref="b")]),
                         "ring": Line(name="ring", items=[LineItem(ref="cell", repeat=2),
                                                          LineItem(ref="cell", reverse=True)])},
                  use="ring")
    fl = lat.flatten()
    assert [p.name for p in fl] == ["d", "b", "d", "b", "b", "d"]
    assert fl[-2].reversed and not fl[0].reversed
    assert fl[-1].s_out == pytest.approx(6.0) and lat.total_length == 6.0
    assert fl[2].path == ("ring", "cell")


def test_from_sequence_uniquifies_names():
    lat = Lattice.from_sequence("seq", [Drift(name="d", length=1.0), Drift(name="d", length=2.0)], PROTON)
    assert sorted(lat.elements) == ["d", "d_2"]
    assert [p.length for p in lat.flatten()] == [1.0, 2.0]


def test_walk_energy_rule():
    cav = RFCavity(name="c", rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30), frequency_Hz=162.5e6))
    lat = Lattice.from_sequence("l", [Drift(name="d1", length=0.5), Freq(name="f", frequency_Hz=325e6), cav,
                                      ReferenceChange(name="rc", dE_ref_eV=-0.1e6), Drift(name="d2", length=0.5)],
                                PROTON)
    pl = propagate(lat)
    assert pl[2].ref_in.kinetic_energy_eV == 2.1e6
    assert pl[2].ref_out.kinetic_energy_eV == pytest.approx(2.1e6 + 0.8660254e6)
    assert pl[2].ref_in.rf_frequency_Hz == 325e6            # FREQ switched the clock
    assert pl[3].ref_out.kinetic_energy_eV == pytest.approx(2.1e6 + 0.8660254e6 - 0.1e6)
    assert pl[-1].ref_out.time_s > pl[0].ref_out.time_s
    # species independence of the IR convention
    lat_h = lat.model_copy(update={"reference": PROTON.model_copy(update={"species": species("h-")})})
    assert propagate(lat_h)[2].ref_out.kinetic_energy_eV == pytest.approx(pl[2].ref_out.kinetic_energy_eV)


def test_normalize_roundtrip_sign():
    r = ReferenceParticle(species=species("h-"), kinetic_energy_eV=800e6)
    assert k1_from_gradient(gradient_from_k1(0.6, r), r) == pytest.approx(0.6)
    assert gradient_from_k1(0.6, r) < 0            # H-: positive K1 is a negative lab gradient


def test_rf_phase_conventions():
    phi = math.radians(-30)
    assert rfconv.madx_lag(phi) == pytest.approx(0.25 - 30 / 360)
    assert rfconv.phase_from_madx_lag(rfconv.madx_lag(phi)) == pytest.approx(phi)
    assert rfconv.elegant_phase_deg(phi, -1) == pytest.approx(60.0)
    assert rfconv.elegant_phase_deg(phi, +1) == pytest.approx(240.0)
    assert rfconv.phase_from_elegant_deg(240.0, +1) == pytest.approx(phi)
    assert rfconv.bmad_phi0(phi) == pytest.approx(-30 / 360)
    assert rfconv.tracewin_phase_deg(phi, -1, sync=True) == pytest.approx(-30.0)
    assert rfconv.tracewin_phase_deg(phi, -1, sync=False) == pytest.approx(150.0)
    assert rfconv.phase_from_tracewin_deg(150.0, -1, sync=False) == pytest.approx(phi)


def test_survey_straight_and_bend():
    lat = Lattice.from_sequence("l", [Drift(name="d", length=2.0),
                                      Bend(name="b", length=1.0, bend=BendP(angle=math.pi / 2)),
                                      Drift(name="d2", length=1.0)], PROTON)
    sv = survey(lat.flatten())
    np.testing.assert_allclose(sv[0], [0, 0, 2.0, 0])
    rho = 1.0 / (math.pi / 2)
    np.testing.assert_allclose(sv[1], [-rho, 0, 2.0 + rho, -math.pi / 2], atol=1e-12)
    np.testing.assert_allclose(sv[2], [-rho - 1.0, 0, 2.0 + rho, -math.pi / 2], atol=1e-12)


def test_fidelity_report_strict_and_allowlist():
    rep = FidelityReport(source_format="tracewin", target_format="madx")
    rep.exact("q1", "Quadrupole")
    rep.equivalent("CONST_P0", "MAD-X keeps p0", element="c1", kind="RFCavity")
    rep.lossy("FM_TO_CAVITY", "field map degraded", element="fm1", kind="FieldMap")
    assert rep.counts == {"EXACT": 1, "EQUIVALENT": 1, "LOSSY": 1}
    with pytest.raises(TranslationError):
        rep.raise_if(True)
    rep.allowlist.add("FM_TO_CAVITY")
    rep.raise_if(True)
    assert "FM_TO_CAVITY" in rep.summary()
    assert FidelityClass("LOSSY") is FidelityClass.LOSSY


def test_element_roundtrip_dict():
    q = Quadrupole(name="qf", length=0.3)
    q.gradient = 2.5
    lat = Lattice.from_sequence("l", [q], PROTON)
    lat2 = Lattice.from_dict(lat.to_dict())
    assert lat2.elements["qf"].gradient == 2.5 and lat2.flatten()[0].length == 0.3
    assert len(ALL_KINDS) == 22
