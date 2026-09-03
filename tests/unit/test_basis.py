"""Engine-free checks of the coordinate-basis machinery."""
from __future__ import annotations

import numpy as np
import pytest

from lattix.oracles.base import SPECIES, Basis, BeamSpec, OracleResult
from lattix.oracles.basis import (
    drift_common,
    momentum_eV,
    rescale_to_constant_p0,
    transform_matrix,
)

MP = SPECIES["proton"][0]


@pytest.mark.parametrize("basis", [b for b in Basis if b is not Basis.COMMON])
def test_transform_is_diagonal_and_invertible(basis):
    T = transform_matrix(basis, 2.1e6, MP, rf_frequency_Hz=162.5e6)
    assert T.shape == (6, 6)
    assert np.allclose(T, np.diag(np.diag(T)))
    assert np.all(np.diag(T) != 0)


def test_common_is_identity():
    assert np.allclose(transform_matrix(Basis.COMMON, 1e6, MP), np.eye(6))


def test_helix_needs_frequency():
    with pytest.raises(ValueError):
        transform_matrix(Basis.HELIX, 2.1e6, MP)


def test_helix_scaling_matches_tracewin_crosscheck_convention():
    """Inverse of HELIX's diag(-360/(βλ), β²γ·mc²) at test_tracewin_crosscheck.py:325."""
    beam = BeamSpec("proton", 2.1e6, 162.5e6)
    T = transform_matrix(Basis.HELIX, beam.kinetic_energy_eV, MP, 162.5e6)
    lam = 299_792_458.0 / 162.5e6
    assert T[4, 4] == pytest.approx(-beam.beta * lam / 360.0, rel=1e-12)
    assert T[5, 5] == pytest.approx(1e6 / (beam.beta**2 * beam.gamma * MP), rel=1e-12)
    assert T[0, 0] == T[1, 1] == 1e-3


def test_drift_common_r56_is_l_over_gamma2():
    beam = BeamSpec("proton", 800e6)
    R = drift_common(2.0, 800e6, MP)
    assert R[0, 1] == R[2, 3] == 2.0
    assert R[4, 5] == pytest.approx(2.0 / beam.gamma**2)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rescale_to_constant_p0_makes_transverse_det_one():
    p_in, p_out = 1.0, 1.2
    R = np.eye(6)
    R[1, 1] = R[3, 3] = R[5, 5] = p_in / p_out    # adiabatic damping of a p0-following map
    S = rescale_to_constant_p0(R, p_in, p_out)
    assert np.linalg.det(S[:2, :2]) == pytest.approx(1.0)
    assert np.linalg.det(S[2:4, 2:4]) == pytest.approx(1.0)
    assert np.allclose(rescale_to_constant_p0(R, 1.0, 1.0), R)


def test_momentum_and_beamspec_consistent():
    b = BeamSpec("h-", 2.1e6)
    assert momentum_eV(2.1e6, b.mass_eV) == pytest.approx(b.pc_eV, rel=1e-12)
    assert b.charge == -1 and b.mass_eV > MP


def _fake(engine, basis, names, lengths, R, ke_in, ke_out, f=None):
    n = len(names)
    return OracleResult(engine=engine, basis=basis, names=names, length=np.array(lengths),
                        s_out=np.cumsum(lengths), R_elem=np.array(R),
                        ref_kinetic_eV_in=np.array(ke_in), ref_kinetic_eV_out=np.array(ke_out),
                        mass_eV=MP, charge=1, rf_frequency_Hz=None if f is None else np.full(n, f))


def test_to_common_roundtrip_drift_in_helix_basis():
    """A HELIX-basis drift (mm/mrad/deg/MeV) maps onto the analytic common drift."""
    ke, f = 2.1e6, 162.5e6
    T = transform_matrix(Basis.HELIX, ke, MP, f)
    Rc = drift_common(1.0, ke, MP)
    R_helix = np.linalg.inv(T) @ Rc @ T
    res = _fake("x", Basis.HELIX, ["d"], [1.0], [R_helix], [ke], [ke], f)
    assert np.allclose(res.to_common().R_elem[0], Rc, atol=1e-12)


def test_result_validates_shapes():
    with pytest.raises(ValueError):
        _fake("x", Basis.COMMON, ["a", "b"], [1.0], [np.eye(6)], [1.0], [1.0])


def test_r_cum_is_product():
    A = np.eye(6)
    A[0, 1] = 1.0
    B = np.eye(6)
    B[1, 0] = -0.5
    res = _fake("x", Basis.COMMON, ["a", "b"], [1.0, 0.0], [A, B], [1e6, 1e6], [1e6, 1e6])
    assert np.allclose(res.R_cum[1], B @ A)
