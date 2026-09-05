"""Coordinate-basis transforms onto the common basis.

Common basis ``C`` = (x [m], px/p0, y [m], py/p0, z [m], δ) with z *ahead-positive*
(a particle ahead of the reference has z > 0) and δ = Δp/p0.

Every transform is linear and diagonal (first order), evaluated with the local
(β, γ) of the reference particle at the boundary where it is applied.  The
longitudinal *signs* are deliberately parameters (``_T_SIGN``, ``_E_SCALE``):
PLAN §5.1 forbids trusting a remembered convention, so ``tests/oracles``
fingerprints each engine (1 m drift R56, thin-cavity gain) and the values
below are whatever those measurements pinned.
"""
from __future__ import annotations

import numpy as np

from lattix.oracles.base import C_LIGHT, Basis

# native longitudinal position coordinate -> z_common = _Z_SIGN[basis] * scale * q5
# (scale: 1 for length-like coordinates; -βλ/360 for TraceWin/HELIX phase in deg.)
_Z_SIGN: dict[Basis, float] = {
    Basis.MADX: +1.0,      # MEASURED 2026-09-03 (cpymad 5.09.03 drift R56 > 0): T ahead-positive
    Basis.IMPACTX: +1.0,   # same canonical set as MAD-X (unverified until the impactx fingerprint runs)
    Basis.XTRACK: +1.0,    # zeta = s - β c t (ahead positive)
    Basis.BMAD: +1.0,      # z = -β c Δt (ahead positive)
    Basis.ELEGANT: -1.0,   # s = path length / arrival (late positive)      [fingerprinted]
    Basis.TRACEWIN: +1.0,  # z ahead positive (HELIX test_tracewin_crosscheck.py:325-327)
    Basis.HELIX: -1.0,     # Δφ [deg]: late positive -> z = -(βλ/360)·Δφ
    Basis.FLAME: -1.0,     # φ [rad]: late positive
    Basis.IMPACTZ: -1.0,   # Δφ [rad]
    Basis.CHEETAH: -1.0,   # MEASURED 2026-09-05 (Cheetah 0.8.4 drift R56 = −L/(β²γ²)): τ = c·Δt late-positive
    Basis.PYORBIT: +1.0,   # MEASURED 2026-09-05 (PyORBIT3 drift R56 = +L/γ² · 1e9/(β²γ mc²)): z ahead-positive
    Basis.OCELOT: -1.0,    # MEASURED 2026-09-05 (Ocelot 25.06 drift R56 = −L/(β²γ²)): τ = c·Δt late-positive
}


def _beta_gamma(kinetic_eV: float, mass_eV: float) -> tuple[float, float]:
    g = 1.0 + kinetic_eV / mass_eV
    b = float(np.sqrt(1.0 - 1.0 / (g * g)))
    return b, g


def transform_matrix(basis: Basis, kinetic_eV: float, mass_eV: float,
                     rf_frequency_Hz: float | None = None, mass_per_nucleon_eV: float | None = None,
                     ) -> np.ndarray:
    """6×6 ``T`` such that ``x_common = T @ x_native`` at a boundary with the given
    reference kinetic energy."""
    if basis is Basis.COMMON:
        return np.eye(6)
    beta, gamma = _beta_gamma(kinetic_eV, mass_eV)
    d = np.ones(6)
    zs = _Z_SIGN[basis]
    if basis in (Basis.MADX, Basis.IMPACTX, Basis.CHEETAH, Basis.OCELOT):
        # (x, px, y, py, t, pt): px = p_x/p0 ; z = zs * β t ; δ ≈ pt/β
        d[4] = zs * beta
        d[5] = 1.0 / beta
    elif basis in (Basis.XTRACK, Basis.BMAD, Basis.TRACEWIN):
        d[4] = zs
    elif basis is Basis.ELEGANT:
        # (x, x', y, y', s, δ): x' ≈ px/p0 to first order
        d[4] = zs
    elif basis is Basis.PYORBIT:
        # (x, x', y, y', z [m], dE [GeV]): δ = ΔE/(β² E_tot) = dE·1e9/(β²γ m c²)
        d[4] = zs
        d[5] = 1e9 / (beta * beta * gamma * mass_eV)
    elif basis is Basis.HELIX:
        # (mm, mrad, mm, mrad, deg, MeV)
        if not rf_frequency_Hz:
            raise ValueError("HELIX basis needs the RF frequency to convert phase to length")
        lam = C_LIGHT / rf_frequency_Hz
        d[0] = d[1] = d[2] = d[3] = 1e-3
        d[4] = zs * beta * lam / 360.0
        d[5] = 1e6 / (beta * beta * gamma * mass_eV)  # ΔW[MeV] -> δ = ΔW/(β²γ m c²)
    elif basis is Basis.FLAME:
        if not rf_frequency_Hz:
            raise ValueError("FLAME basis needs the RF frequency to convert phase to length")
        lam = C_LIGHT / rf_frequency_Hz
        mpn = mass_per_nucleon_eV or mass_eV
        d[0] = d[2] = 1e-3
        d[4] = zs * beta * lam / (2 * np.pi)
        d[5] = 1e6 * (mass_eV / mpn) / (beta * beta * gamma * mass_eV)
    elif basis is Basis.IMPACTZ:
        # MEASURED 2026-09-03 (impact-z 2.7.7): native (x/Scxl, γβx, y/Scxl, γβy, ω(t−t_ref) [rad,
        # late-positive], γ_ref−γ) with Scxl = c/(2πf); a 1 m drift then gives R56_common = +0.9955387
        if not rf_frequency_Hz:
            raise ValueError("IMPACT-Z basis needs the RF frequency")
        lam = C_LIGHT / rf_frequency_Hz
        xl = lam / (2 * np.pi)
        bg = beta * gamma
        d[0] = d[2] = xl
        d[1] = d[3] = 1.0 / bg               # γβ_x -> px/p0
        d[4] = zs * beta * xl                # phase [rad] -> z
        d[5] = -1.0 / (beta * beta * gamma)  # γ_ref − γ  ->  δ (sign: energy DEFICIT is positive)
    else:  # pragma: no cover
        raise NotImplementedError(basis)
    return np.diag(d)


def drift_common(length: float, kinetic_eV: float, mass_eV: float) -> np.ndarray:
    """Analytic drift map in the common basis: R56 = L/γ² (z ahead-positive, δ = Δp/p)."""
    _, gamma = _beta_gamma(kinetic_eV, mass_eV)
    R = np.eye(6)
    R[0, 1] = length
    R[2, 3] = length
    R[4, 5] = length / (gamma * gamma)
    return R


def damping_normalised(R: np.ndarray) -> np.ndarray:
    """Divide each transverse 2×2 block by sqrt(det) so p0-following engines
    (Bmad lcavity, Elegant change_p0, TraceWin) can be compared with constant-p0
    engines (MAD-X, xtrack) through accelerating elements."""
    out = R.copy()
    for a in (0, 2):
        blk = R[a:a + 2, a:a + 2]
        det = np.linalg.det(blk)
        if det > 0:
            out[a:a + 2, :] /= np.sqrt(det)
    return out


def rescale_to_constant_p0(R: np.ndarray, p_in: float, p_out: float) -> np.ndarray:
    """Express a p0-following map (momentum coordinates normalised to the local
    p0 at each end) in constant-p0 coordinates (everything normalised to p_in):
    scale the momentum-like rows (px, py, δ) by p_out/p_in.  Exact for linear
    maps; makes det(transverse 2×2) = 1 so p0-following engines (Bmad lcavity,
    Elegant change_p0, TraceWin, HELIX) compare directly with constant-p0
    engines (MAD-X, xtrack).  Identity when p_in == p_out."""
    out = R.copy()
    f = p_out / p_in
    for row in (1, 3, 5):
        out[row, :] *= f
    return out


def momentum_eV(kinetic_eV: float, mass_eV: float) -> float:
    g = 1.0 + kinetic_eV / mass_eV
    return float(np.sqrt(g * g - 1.0) * mass_eV)
