"""xtrack oracle: basis fingerprint (1 m drift) and self-consistency on a MAD-X FODO.

Every number here is *measured* from the installed xtrack (PLAN §5.1: never
trust a remembered longitudinal sign).  Skips when xtrack is not importable.
"""
from __future__ import annotations

import numpy as np
import pytest

from lattix.oracles import SPECIES, Basis, BeamSpec, Probe, get_oracle
from lattix.oracles.basis import drift_common
from lattix.testing import require

pytestmark = pytest.mark.oracle_xtrack
require("xtrack")

KE = 2.1e6
BEAM = BeamSpec(species="proton", kinetic_energy_eV=KE)
H = 1e-5

FODO = """\
beam, particle=proton, energy=0.94037208816;
qf: quadrupole, l=0.3, k1=0.6;
qd: quadrupole, l=0.3, k1=-0.6;
d1: drift, l=0.5;
m1: marker;
fodo: sequence, refer=centre, l=1.6;
qf, at=0.15;
d1, at=0.55;
m1, at=0.8;
qd, at=0.95;
endsequence;
"""
DRIFT = """\
beam, particle=proton, energy=0.94037208816;
d1: drift, l=1.0;
one: sequence, refer=centre, l=1.0;
d1, at=0.5;
endsequence;
"""


def fd_probe(h: float = H) -> np.ndarray:
    """12 rows: reference ± h in each of the six coordinates."""
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
    deck = tmp_path_factory.mktemp("xtrack") / "fodo.madx"
    deck.write_text(FODO)
    return get_oracle("xtrack").run(deck, beam=BEAM, probe=Probe(coords=fd_probe()))


@pytest.fixture(scope="module")
def drift(tmp_path_factory):
    deck = tmp_path_factory.mktemp("xtrack") / "drift.madx"
    deck.write_text(DRIFT)
    return get_oracle("xtrack").run(deck, beam=BEAM)


def test_structure(fodo):
    assert fodo.engine == "xtrack" and fodo.basis is Basis.XTRACK
    assert not any(n.endswith(("$start", "$end")) for n in fodo.names)
    user = [n for n in fodo.names if not n.startswith("drift")]
    assert user == ["qf", "d1", "m1", "qd"]          # auto drift fills 1.1..1.6
    assert fodo.length[fodo.names.index("m1")] == 0.0
    assert fodo.s_out[-1] == pytest.approx(1.6, abs=1e-12)
    assert fodo.total_length == pytest.approx(1.6, abs=1e-12)
    assert np.all(np.diff(fodo.s_out) >= 0)
    assert not fodo.warnings, fodo.warnings


def test_energy_is_constant(fodo):
    assert fodo.mass_eV == SPECIES["proton"][0] and fodo.charge == 1
    assert np.all(fodo.ref_kinetic_eV_in == KE) and np.all(fodo.ref_kinetic_eV_out == KE)
    assert "constant" in fodo.meta["p0_model"]


def test_product_of_element_maps_matches_full_line_finite_difference(fodo):
    R_full = fd_from_probe(fd_probe(), fodo.probe_out)   # one track through the whole line
    R_prod = fodo.R_cum[-1]
    err = np.max(np.abs(R_prod - R_full))
    assert err < 1e-8, f"max |prod(R_elem) - R_fd| = {err:.3e}\nprod:\n{R_prod}\nfd:\n{R_full}"


def test_every_element_map_is_symplectic_to_1e8(fodo):
    dets = np.array([np.linalg.det(R) for R in fodo.R_elem])
    assert np.all(np.isfinite(fodo.R_elem))
    assert np.max(np.abs(dets - 1)) < 1e-8, dict(zip(fodo.names, dets, strict=True))
    for R in fodo.R_elem:                     # transverse blocks individually
        for a in (0, 2):
            assert abs(np.linalg.det(R[a:a + 2, a:a + 2]) - 1) < 1e-8


def test_drift_basis_fingerprint(drift):
    """1 m drift: common-basis map must be the analytic one (R56 = +L/γ², ahead-positive z)."""
    assert drift.names == ["d1"] and drift.length[0] == pytest.approx(1.0)
    native = drift.R_elem[0]
    common = drift.to_common().R_elem[0]
    expected = drift_common(1.0, drift.ref_kinetic_eV_in[0], drift.mass_eV)
    diff = np.max(np.abs(common - expected))
    assert diff < 1e-9, (
        f"xtrack drift fingerprint mismatch (max |R_common - expected| = {diff:.3e}):\n"
        f"  native R56 (zeta,delta) = {native[4, 5]!r}\n"
        f"  common R[4,5]           = {common[4, 5]!r}\n"
        f"  expected L/gamma^2      = {expected[4, 5]!r}\n"
        f"common:\n{common}\nexpected:\n{expected}")


def test_survey_end_point(fodo):
    X, Y, Z, theta = fodo.survey[-1]
    assert Z == pytest.approx(1.6, abs=1e-12)
    assert abs(X) < 1e-12 and abs(Y) < 1e-12 and abs(theta) < 1e-12
    assert np.allclose(fodo.survey[:, 2], fodo.s_out, atol=1e-12)


def test_twiss_rows_are_element_exits(fodo):
    """β at the exit of the first quad, propagated through its measured map, must
    match the twiss table row the adapter picked for that exit."""
    R = fodo.R_elem[0]
    b0, a0 = BEAM.betx, BEAM.alfx
    g0 = (1 + a0 * a0) / b0
    betx_exit = R[0, 0] ** 2 * b0 - 2 * R[0, 0] * R[0, 1] * a0 + R[0, 1] ** 2 * g0
    assert fodo.twiss["betx"][0] == pytest.approx(betx_exit, rel=1e-7)
    assert fodo.twiss["betx"][0] != pytest.approx(b0, rel=1e-3)     # i.e. not the entrance row
    Rc = fodo.R_cum[-1]
    bety_end = Rc[2, 2] ** 2 * BEAM.bety + Rc[2, 3] ** 2 / BEAM.bety
    assert fodo.twiss["bety"][-1] == pytest.approx(bety_end, rel=1e-7)
    assert set(fodo.disp) == {"dx", "dpx", "dy", "dpy"}
    assert np.allclose(fodo.disp["dx"], 0.0, atol=1e-9)


def test_default_probe_is_linear(tmp_path):
    deck = tmp_path / "fodo.madx"
    deck.write_text(FODO)
    probe = Probe.default()
    res = get_oracle("xtrack").run(deck, beam=BEAM, probe=probe)
    assert res.probe_out.shape == (64, 6) and np.all(np.isfinite(res.probe_out))
    linear = probe.coords @ res.R_cum[-1].T
    assert np.max(np.abs(res.probe_out - linear)) < 5e-7


def test_deck_beam_used_when_no_beamspec(tmp_path):
    deck = tmp_path / "fodo.madx"
    deck.write_text(FODO)
    res = get_oracle("xtrack").run(deck)
    assert res.ref_kinetic_eV_in[0] == pytest.approx(KE, rel=1e-6)
    assert res.mass_eV == pytest.approx(SPECIES["proton"][0], rel=1e-9)
