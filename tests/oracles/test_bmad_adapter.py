"""Bmad/Tao oracle (via bmad_worker in the ``bmad`` conda env): drift basis
fingerprint, FODO self-consistency, lcavity reference-energy following.

Skips when no interpreter with pytao can be resolved (see lattix.oracles.pytao).
"""
from __future__ import annotations

import numpy as np
import pytest

from lattix.oracles import SPECIES, Basis, BeamSpec, Probe, get_oracle
from lattix.oracles.basis import drift_common, momentum_eV
from lattix.testing import require

pytestmark = pytest.mark.oracle_bmad
require("bmad")

KE = 2.1e6
BEAM = BeamSpec(species="proton", kinetic_energy_eV=KE)
H = 1e-5

FODO = """\
parameter[particle] = proton
parameter[e_tot] = 1.738272e9      ! 800 MeV kinetic: the BeamSpec must override this
parameter[geometry] = open
d1: drift, l = 0.5
qf: quadrupole, l = 0.3, k1 = 0.6
qd: quadrupole, l = 0.3, k1 = -0.6
m1: marker
fodo: line = (qf, d1, m1, qd, d1)
use, fodo
"""
DRIFT = """\
parameter[particle] = proton
parameter[e_tot] = 1.738272e9
parameter[geometry] = open
d1: drift, l = 1.0
one: line = (d1)
use, one
"""
CAVITY = """\
parameter[particle] = proton
parameter[e_tot] = 1.738272e9
parameter[geometry] = open
d1: drift, l = 0.5
cav: lcavity, l = 0.01, voltage = 1e6, phi0 = 0, rf_frequency = 162.5e6
acc: line = (d1, cav, d1)
use, acc
"""


def fd_probe(h: float = H) -> np.ndarray:
    P = np.zeros((12, 6))
    for k in range(6):
        P[2 * k, k] = h
        P[2 * k + 1, k] = -h
    return P


def fd_from_probe(P_in: np.ndarray, P_out: np.ndarray) -> np.ndarray:
    R = np.empty((6, 6))
    for k in range(6):
        R[:, k] = (P_out[2 * k] - P_out[2 * k + 1]) / (P_in[2 * k, k] - P_in[2 * k + 1, k])
    return R


@pytest.fixture(scope="module")
def fodo(tmp_path_factory):
    deck = tmp_path_factory.mktemp("bmad") / "fodo.bmad"
    deck.write_text(FODO)
    return get_oracle("bmad").run(deck, beam=BEAM, probe=Probe(coords=fd_probe()))


@pytest.fixture(scope="module")
def drift(tmp_path_factory):
    deck = tmp_path_factory.mktemp("bmad") / "drift.bmad"
    deck.write_text(DRIFT)
    return get_oracle("bmad").run(deck, beam=BEAM)


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    deck = tmp_path_factory.mktemp("bmad") / "cav.bmad"
    deck.write_text(CAVITY)
    return get_oracle("bmad").run(deck, beam=BEAM)


def test_available_reports_interpreter():
    ok, why = get_oracle("bmad").available()
    assert ok and "pytao" in why


def test_structure(fodo):
    assert fodo.engine == "bmad" and fodo.basis is Basis.BMAD
    assert fodo.names == ["QF", "D1", "M1", "QD", "D1", "END"]   # BEGINNING skipped, END kept
    assert fodo.length.tolist() == pytest.approx([0.3, 0.5, 0.0, 0.3, 0.5, 0.0])
    assert fodo.s_out[-1] == pytest.approx(1.6, abs=1e-12)
    assert fodo.total_length == pytest.approx(1.6, abs=1e-12)
    assert fodo.meta["tao_version"] and fodo.meta["python"]
    assert not fodo.warnings, fodo.warnings


def test_beamspec_overrides_deck_energy(fodo):
    """The deck says 800 MeV; the BeamSpec (2.1 MeV) must win, in Bmad's own mass."""
    assert fodo.charge == 1
    assert fodo.mass_eV == pytest.approx(SPECIES["proton"][0], rel=1e-8)   # 938272089.43 vs .16
    assert np.allclose(fodo.ref_kinetic_eV_in, KE, atol=1e-3)
    assert np.allclose(fodo.ref_kinetic_eV_out, KE, atol=1e-3)


def test_product_of_element_maps_matches_full_line_finite_difference(fodo):
    R_full = fd_from_probe(fd_probe(), fodo.probe_out)   # Bmad *tracking* of ± h probes
    R_prod = fodo.R_cum[-1]                              # product of Bmad's ele_mat6
    err = np.max(np.abs(R_prod - R_full))
    assert err < 1e-8, f"max |prod(mat6) - R_fd| = {err:.3e}\nprod:\n{R_prod}\nfd:\n{R_full}"


def test_every_element_map_is_symplectic_to_1e8(fodo):
    dets = np.array([np.linalg.det(R) for R in fodo.R_elem])
    assert np.max(np.abs(dets - 1)) < 1e-8, dict(zip(fodo.names, dets, strict=True))


def test_drift_basis_fingerprint(drift):
    """1 m drift: common-basis map must be the analytic one (R56 = +L/γ², ahead-positive z)."""
    assert drift.names == ["D1", "END"]
    native = drift.R_elem[0]
    common = drift.to_common().R_elem[0]
    expected = drift_common(1.0, drift.ref_kinetic_eV_in[0], drift.mass_eV)
    diff = np.max(np.abs(common - expected))
    assert diff < 1e-9, (
        f"Bmad drift fingerprint mismatch (max |R_common - expected| = {diff:.3e}):\n"
        f"  native R56 (z,pz)  = {native[4, 5]!r}\n"
        f"  common R[4,5]      = {common[4, 5]!r}\n"
        f"  expected L/gamma^2 = {expected[4, 5]!r}\n"
        f"common:\n{common}\nexpected:\n{expected}")
    assert np.allclose(drift.R_elem[1], np.eye(6), atol=1e-15)   # END marker


def test_survey_end_point(fodo):
    X, Y, Z, theta = fodo.survey[-1]
    assert Z == pytest.approx(1.6, abs=1e-12)
    assert abs(X) < 1e-12 and abs(Y) < 1e-12 and abs(theta) < 1e-12
    assert np.allclose(fodo.survey[:, 2], fodo.s_out, atol=1e-12)


def test_twiss_rows_are_element_exits(fodo):
    R = fodo.R_elem[0]
    b0, a0 = BEAM.betx, BEAM.alfx
    g0 = (1 + a0 * a0) / b0
    betx_exit = R[0, 0] ** 2 * b0 - 2 * R[0, 0] * R[0, 1] * a0 + R[0, 1] ** 2 * g0
    assert fodo.twiss["betx"][0] == pytest.approx(betx_exit, rel=1e-7)
    assert fodo.twiss["betx"][0] != pytest.approx(b0, rel=1e-3)
    Rc = fodo.R_cum[-1]
    bety_end = Rc[2, 2] ** 2 * BEAM.bety + Rc[2, 3] ** 2 / BEAM.bety
    assert fodo.twiss["bety"][-1] == pytest.approx(bety_end, rel=1e-7)
    assert np.allclose(fodo.disp["dx"], 0.0, atol=1e-9)


def test_lcavity_follows_reference_momentum(cavity):
    assert cavity.names == ["D1", "CAV", "D1", "END"]
    gain = cavity.ref_kinetic_eV_out - cavity.ref_kinetic_eV_in
    i = cavity.names.index("CAV")
    assert gain[i] == pytest.approx(1e6, rel=1e-3), f"measured lcavity gain {gain[i]!r} eV"
    assert np.allclose(np.delete(gain, i), 0.0, atol=1e-6)
    assert cavity.ref_kinetic_eV_in[i + 1] == cavity.ref_kinetic_eV_out[i]
    # p0-following: transverse 2x2 determinants equal p_in/p_out (adiabatic damping)
    p_ratio = (momentum_eV(cavity.ref_kinetic_eV_in[i], cavity.mass_eV)
               / momentum_eV(cavity.ref_kinetic_eV_out[i], cavity.mass_eV))
    R = cavity.R_elem[i]
    for a in (0, 2):
        assert np.linalg.det(R[a:a + 2, a:a + 2]) == pytest.approx(p_ratio, rel=1e-5)
    assert "follows p0" in cavity.meta["p0_model"]


def test_default_probe_is_linear(tmp_path):
    deck = tmp_path / "fodo.bmad"
    deck.write_text(FODO)
    probe = Probe.default()
    res = get_oracle("bmad").run(deck, beam=BEAM, probe=probe)
    assert res.probe_out.shape == (64, 6) and np.all(np.isfinite(res.probe_out))
    linear = probe.coords @ res.R_cum[-1].T
    assert np.max(np.abs(res.probe_out - linear)) < 5e-7


def test_deck_energy_used_when_no_beamspec(tmp_path):
    deck = tmp_path / "fodo.bmad"
    deck.write_text(FODO)
    res = get_oracle("bmad").run(deck)
    assert res.ref_kinetic_eV_in[0] == pytest.approx(1.738272e9 - res.mass_eV, rel=1e-9)
