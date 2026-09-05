"""Design check for the IMPACT-T solenoid table written by ``lattix.formats.impactt``.

IMPACT-T tracks a type-3 solenoid from a ``1T<id>.T7`` ``(r, z)`` table with bilinear
interpolation (``Sol.f90`` ``getfldt_Sol``).  This script integrates the paraxial lab-frame
equations of motion through such a table — the model reproduces IMPACT-T's own map to 1e-9 —
and compares the result with the hard-edge solenoid map.  The writer's edge design (three
intervals of width ``h`` straddling each nominal end, ``Bz/B = 0, a, 1 − a, 1`` and
``Br/(B·r/h) = 0, −¼, −¼, 0``) was found by minimising that deviation over ``a``; it is
``O(h²)`` for any length, field, rigidity and sign, whereas a plain ``0 → ½ → 1`` ramp is ``O(h)``.

    python tools/impactt_solenoid_table.py          # prints the deviation for the shipped a and a scan
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from lattix.formats.impactt.writer import _SOL_EDGE_A


def hard_edge(K: float, L: float) -> np.ndarray:
    C, S = math.cos(K * L), math.sin(K * L)
    return np.array([[C * C, S * C / K, S * C, S * S / K], [-K * S * C, C * C, -K * S * S, S * C],
                     [-S * C, -S * S / K, C * C, S * C / K], [K * S * S, -S * C, -K * S * C, C * C]])


def body(K: float, L: float) -> np.ndarray:
    """The solenoid body without its edge kicks (``Bz`` only): a Larmor rotation by ``2KL``."""
    c, s = math.cos(2 * K * L), math.sin(2 * K * L)
    return np.array([[1, s / (2 * K), 0, (1 - c) / (2 * K)], [0, c, 0, s],
                     [0, -(1 - c) / (2 * K), 1, s / (2 * K)], [0, -s, 0, c]])


def drift(length: float) -> np.ndarray:
    return np.array([[1, length, 0, 0], [0, 1, 0, 0], [0, 0, 1, length], [0, 0, 0, 1]])


def _interp(zn, vals, z):
    if z <= zn[0] or z >= zn[-1]:
        return 0.0
    j = min(max(int(np.searchsorted(zn, z) - 1), 0), len(zn) - 2)
    f = (z - zn[j]) / (zn[j + 1] - zn[j])
    return vals[j] * (1 - f) + vals[j + 1] * f


def region_map(zn, bz, br_over_r, brho: float) -> np.ndarray:
    """Paraxial lab-frame map through a piecewise-linear ``Bz(z)``, ``Br = g(z)·r`` table region:
    ``x'' = (y' Bz − g y)/Bρ``, ``y'' = (g x − x' Bz)/Bρ``."""
    def rhs(z, y):
        Bz, g = _interp(zn, bz, z), _interp(zn, br_over_r, z)
        out = []
        for c in range(4):
            x, xp, yy, yp = y[4 * c:4 * c + 4]
            out += [xp, (Bz * yp - g * yy) / brho, yp, (g * x - xp * Bz) / brho]
        return out
    sol = solve_ivp(rhs, (zn[0], zn[-1]), np.eye(4).flatten(), rtol=1e-13, atol=1e-18,
                    max_step=min(np.diff(zn)) / 8)
    return sol.y[:, -1].reshape(4, 4).T


def table_map(L: float, B: float, brho: float, h: float, a: float) -> np.ndarray:
    """Map between the nominal ends of a solenoid written with edge parameter ``a``."""
    edge = [0.0, a, 1.0 - a, 1.0]
    z_in = [(-1.5 + j) * h for j in range(4)]
    m_in = region_map(z_in, [B * v for v in edge], [0.0, -0.25 * B / h, -0.25 * B / h, 0.0], brho)
    z_out = [L + (-1.5 + j) * h for j in range(4)]
    m_out = region_map(z_out, [B * v for v in edge[::-1]], [0.0, 0.25 * B / h, 0.25 * B / h, 0.0], brho)
    return drift(-1.5 * h) @ m_out @ body(B / (2 * brho), L - 3 * h) @ m_in @ drift(-1.5 * h)


def deviation(L: float, B: float, brho: float, h: float, a: float) -> float:
    return float(np.abs(table_map(L, B, brho, h, a) - hard_edge(B / (2 * brho), L)).max())


if __name__ == "__main__":
    L, B, brho = 0.4, 0.5, 0.20951311297823577          # 2.1 MeV protons
    print(f"shipped a = {_SOL_EDGE_A}")
    for h in (1e-4, 2e-4, 4e-4, 1e-3):
        print(f"  h = {h:g}: max |M − hard edge| = {deviation(L, B, brho, h, _SOL_EDGE_A):.2e}")
    for L2, B2, brho2 in ((0.25, 1.0, brho), (0.6, 0.3, 0.6), (0.4, -0.5, brho)):
        print(f"  L={L2} B={B2} Bρ={brho2}: {deviation(L2, B2, brho2, 2e-4, _SOL_EDGE_A):.2e}")
    r = minimize_scalar(lambda a: deviation(L, B, brho, 1e-4, a), bracket=(-0.24, -0.23, -0.22), tol=1e-12)
    print(f"re-optimised a at h = 1e-4: {r.x:.7f} (deviation {r.fun:.2e})")
