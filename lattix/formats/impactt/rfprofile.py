"""On-axis RF profiles for IMPACT-T's type-104 cavity: Fourier coefficients, transit-time
calibration and the synthetic shapes lattix generates.

IMPACT-T (``SC.f90`` ``getfldt_SC``, ``Data.f90`` ``read1t_Data``) reads an ``rfdataN`` file as
one number per line, ``Fcoef(1), Fcoef(2), …``, and reconstructs the on-axis field as

    Ez(z) = Fcoef(1)/2 + Σ_k [Fcoef(2k) cos(2πk·u/L) + Fcoef(2k+1) sin(2πk·u/L)],  u = z − z_mid

with **the element length as the period** and the element centre as the phase reference; the
field is then ``scale·Ez(u)·cos(2πf·t + θ0)`` with ``t`` the absolute simulation time and ``θ0``
the card's phase in degrees (``Param(4)``).  The transverse RF fields follow from the paraxial
expansion of that profile.  Nothing here depends on the rest of lattix.
"""
from __future__ import annotations

import cmath
import math

import numpy as np

C_LIGHT = 299_792_458.0


def raised_cosine(length: float, active: float, n: int) -> tuple[list[float], list[float]]:
    """``Ez ∝ 1 − cos(2πu)`` over the active window centred in ``length``, zero in the flanks
    (the profile and its derivative vanish at both ends, so a hard-edge radial impulse never
    appears).  ``z`` runs from ``−length/2`` to ``+length/2``."""
    n = max(17, int(n) | 1)
    z = [-0.5 * length + length * i / (n - 1) for i in range(n)]
    z0 = -0.5 * active
    ez = []
    for zz in z:
        u = (zz - z0) / active if active > 0 else 0.0
        ez.append(0.0 if u <= 0.0 or u >= 1.0 else 1.0 - math.cos(2.0 * math.pi * u))
    return z, ez


def cell_train(length: float, n_cell: int, *, pi_mode: bool, n: int | None = None,
               active: float | None = None) -> tuple[list[float], list[float]]:
    """``n_cell`` half-wave cells over the ``active`` window (default: the whole length) centred in
    ``length``, zero outside: π mode alternates sign, 2π mode does not."""
    n_cell = max(1, int(n_cell))
    n = max(17, 16 * n_cell + 1) if n is None else max(17, int(n) | 1)
    active = length if not active or active > length else active
    z = [-0.5 * length + length * i / (n - 1) for i in range(n)]
    ez = []
    for zz in z:
        u = (zz + 0.5 * active) / active
        if u <= 0.0 or u >= 1.0:
            ez.append(0.0)
        else:
            ez.append(math.sin(n_cell * math.pi * u) if pi_mode else 1.0 - math.cos(2.0 * n_cell * math.pi * u))
    return z, ez


def fourier_coefficients(z: list[float], ez: list[float], period: float, harmonics: int) -> list[float]:
    """IMPACT-T's ``Fcoef`` list for the sampled profile ``ez(z)`` (``z`` relative to the centre):
    ``[a0, a1, b1, a2, b2, …]`` with ``a_k = (2/L)∫Ez cos(2πk z/L) dz``, ``b_k`` with ``sin``
    (trapezoid rule on the samples, as ``utilities/RFcoeflcls.f90`` does)."""
    n = len(z)
    if n < 2:
        raise ValueError("a profile needs at least two samples")
    out = [0.0] * (2 * harmonics + 1)
    for k in range(harmonics + 1):
        w = 2.0 * math.pi * k / period
        a = b = 0.0
        for i in range(n - 1):
            dz = z[i + 1] - z[i]
            for zz, e in ((z[i], ez[i]), (z[i + 1], ez[i + 1])):
                a += 0.5 * dz * e * math.cos(w * zz)
                b += 0.5 * dz * e * math.sin(w * zz)
        if k == 0:
            out[0] = 2.0 * a / period
        else:
            out[2 * k - 1] = 2.0 * a / period
            out[2 * k] = 2.0 * b / period
    return out


def evaluate_fourier(coefs: list[float], u: float, period: float) -> float:
    """``Ez(u)`` from IMPACT-T's coefficient list (``u`` relative to the element centre)."""
    if not coefs:
        return 0.0
    e = 0.5 * coefs[0]
    n = len(coefs)
    k = 1
    while 2 * k - 1 < n:
        w = 2.0 * math.pi * k * u / period
        e += coefs[2 * k - 1] * math.cos(w)
        if 2 * k < n:
            e += coefs[2 * k] * math.sin(w)
        k += 1
    return e


def evaluate_fourier_array(coefs: list[float], u: np.ndarray, period: float) -> np.ndarray:
    """:func:`evaluate_fourier` on an array of positions."""
    u = np.asarray(u, dtype=float)
    if not coefs:
        return np.zeros_like(u)
    e = np.full_like(u, 0.5 * coefs[0])
    n = len(coefs)
    k = 1
    while 2 * k - 1 < n:
        w = 2.0 * math.pi * k * u / period
        e += coefs[2 * k - 1] * np.cos(w)
        if 2 * k < n:
            e += coefs[2 * k] * np.sin(w)
        k += 1
    return e


def sample_fourier(coefs: list[float], period: float, n: int = 401) -> tuple[list[float], list[float]]:
    """The profile reconstructed on ``n`` points across one period (centre at 0)."""
    n = max(17, int(n) | 1)
    z = [-0.5 * period + period * i / (n - 1) for i in range(n)]
    return z, [evaluate_fourier(coefs, zz, period) for zz in z]


def transit_factor(z: list[float], ez: list[float], k: float) -> complex:
    """``F = ∫ Ez(z) e^{i k z} dz`` (trapezoid; ``z`` relative to the phase reference).  With
    ``k = 2πf/(βc)`` at the reference velocity the energy gain of a particle arriving at the
    centre with RF phase ``ψ`` is ``q·Re[e^{iψ} F]`` to first order in the velocity change."""
    acc = 0j
    for i in range(len(z) - 1):
        dz = z[i + 1] - z[i]
        acc += 0.5 * dz * (ez[i] * cmath.exp(1j * k * z[i]) + ez[i + 1] * cmath.exp(1j * k * z[i + 1]))
    return acc


_CREST_COS = math.cos(0.05)


def _beta(ke_eV: float, mass_eV: float) -> float:
    g = 1.0 + ke_eV / mass_eV
    return math.sqrt(max(1.0 - 1.0 / (g * g), 0.0))


class _Profile:
    """``Ez`` of one card sampled on the RK4 grid (nodes and half nodes over one period)."""

    def __init__(self, coefs: list[float], period: float, frequency_Hz: float, beta: float):
        bl = beta * C_LIGHT / frequency_Hz if (frequency_Hz > 0 and beta > 0) else period
        self.n = int(min(20000, max(96, 64.0 * period / bl)))
        self.period = period
        self.h = period / self.n
        self.ez = evaluate_fourier_array(coefs, np.linspace(-0.5 * period, 0.5 * period, 2 * self.n + 1), period)


def _integrate(prof: _Profile, scale: float, theta0_deg: float, frequency_Hz: float, t_in_s: float,
               ke_in_eV: float, mass_eV: float, charge: float) -> tuple[float, float]:
    """``(ke_out_eV, t_out_s)`` of the reference through the card: RK4 in z on
    ``dW/dz = q·scale·Ez(u)·cos(2πf t + θ0)``, ``dt/dz = 1/(βc)`` from the entrance (time ``t_in``).
    Raises ``ValueError`` when the particle is stopped."""
    w = 2.0 * math.pi * frequency_Hz
    th = math.radians(theta0_deg)
    q = charge * scale
    h, ez = prof.h, prof.ez
    W, t = ke_in_eV, t_in_s

    def rhs(e, W, t):
        g = 1.0 + W / mass_eV
        if g <= 1.0 + 1e-15:
            raise ValueError("the reference particle is stopped inside the cavity")
        return q * e * math.cos(w * t + th), 1.0 / (math.sqrt(1.0 - 1.0 / (g * g)) * C_LIGHT)

    for i in range(prof.n):
        e0, e1, e2 = ez[2 * i], ez[2 * i + 1], ez[2 * i + 2]
        a1, b1 = rhs(e0, W, t)
        a2, b2 = rhs(e1, W + 0.5 * h * a1, t + 0.5 * h * b1)
        a3, b3 = rhs(e1, W + 0.5 * h * a2, t + 0.5 * h * b2)
        a4, b4 = rhs(e2, W + h * a3, t + h * b3)
        W += h / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        t += h / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
    return W, t


def integrate_reference(coefs: list[float], period: float, scale: float, theta0_deg: float, frequency_Hz: float,
                        t_in_s: float, ke_in_eV: float, mass_eV: float, charge: float) -> tuple[float, float]:
    """``(ke_out_eV, t_out_s)`` of the reference particle through a type-104 card (field
    ``scale·Ez(u)·cos(2πf t + θ0)`` with the absolute time ``t``, entering at ``t_in``)."""
    prof = _Profile(coefs, period, frequency_Hz, _beta(ke_in_eV, mass_eV))
    return _integrate(prof, scale, theta0_deg, frequency_Hz, t_in_s, ke_in_eV, mass_eV, charge)


def _crest(prof, scale, f, t_in, ke, mass, charge) -> tuple[float, float]:
    """``(V, θ*)``: the largest gain over the driven phase and the phase giving it (the gain is a
    near-sinusoid of θ0: a three-point fit, then parabolic refinement)."""
    def gain(th):
        try:
            return _integrate(prof, scale, th, f, t_in, ke, mass, charge)[0] - ke
        except ValueError:
            return -math.inf
    g = [gain(th) for th in (0.0, 120.0, 240.0)]
    if any(not math.isfinite(v) for v in g):
        ths = [15.0 * i for i in range(24)]
        g = [gain(th) for th in ths]
        best = max(range(24), key=lambda i: g[i])
        th = ths[best]
    else:
        a = (2.0 / 3.0) * sum(v * math.cos(math.radians(t)) for v, t in zip(g, (0.0, 120.0, 240.0), strict=True))
        b = (2.0 / 3.0) * sum(v * math.sin(math.radians(t)) for v, t in zip(g, (0.0, 120.0, 240.0), strict=True))
        th = math.degrees(math.atan2(b, a))
    for d in (5.0, 0.5, 0.05, 0.005):
        gm, g0, gp = gain(th - d), gain(th), gain(th + d)
        den = gm - 2.0 * g0 + gp
        if den < 0 and all(math.isfinite(v) for v in (gm, g0, gp)):
            th += 0.5 * d * (gm - gp) / den
    return gain(th), (th + 180.0) % 360.0 - 180.0


def gain_from_profile(coefs: list[float], period: float, scale: float, theta0_deg: float, frequency_Hz: float,
                      t_in_s: float, ke_in_eV: float, mass_eV: float, charge: float) -> tuple[float, float, float]:
    """``(dE_eV, V_eV, phase_sync_rad)`` of a type-104 card for the reference particle entering it
    at ``t_in_s`` with kinetic energy ``ke_in_eV``: the gain is integrated through the profile,
    ``V`` is the largest gain over the driven phase and ``dE = V·cos(phase_sync)`` with the sign of
    the phase from the slope (a later particle gains more at a negative phase)."""
    prof = _Profile(coefs, period, frequency_Hz, _beta(ke_in_eV, mass_eV))
    dE = _integrate(prof, scale, theta0_deg, frequency_Hz, t_in_s, ke_in_eV, mass_eV, charge)[0] - ke_in_eV
    v, _ = _crest(prof, scale, frequency_Hz, t_in_s, ke_in_eV, mass_eV, charge)
    if v <= 0:
        return dE, abs(dE), (0.0 if dE >= 0 else math.pi)
    d = 1e-3
    slope = (_integrate(prof, scale, theta0_deg + d, frequency_Hz, t_in_s, ke_in_eV, mass_eV, charge)[0]
             - _integrate(prof, scale, theta0_deg - d, frequency_Hz, t_in_s, ke_in_eV, mass_eV, charge)[0]) \
        / (2.0 * math.radians(d))
    c = min(1.0, max(-1.0, dE / v))
    s = -slope / v
    # the gain fixes cos(phase) exactly and the slope its sign; within 0.05 rad of the crest (or the
    # trough) acos is ill-conditioned and the slope, ∝ sin(phase), is the better measure
    if abs(c) < _CREST_COS:
        psi = math.copysign(math.acos(c), s) if s else (0.0 if c >= 0 else math.pi)
    else:
        psi = math.atan2(s, c)
    return dE, v, psi


def calibrate(coefs: list[float], period: float, voltage_V: float, phase_sync_rad: float, frequency_Hz: float,
              t_in_s: float, ke_in_eV: float, mass_eV: float, charge: float) -> tuple[float, float]:
    """The inverse of :func:`gain_from_profile`: ``(scale [V/m], theta0 [deg])`` such that the
    largest gain over the driven phase is ``voltage_V`` and the reference gains
    ``V·cos(phase_sync)`` on the branch where a later particle gains more for a negative phase."""
    beta = _beta(ke_in_eV, mass_eV)
    prof = _Profile(coefs, period, frequency_Hz, beta)
    z, ez = sample_fourier(coefs, period, 401)
    f = abs(transit_factor(z, ez, 2.0 * math.pi * frequency_Hz / (beta * C_LIGHT)))
    if f < 1e-300:
        raise ValueError("the RF profile integrates to zero at the reference velocity")
    scale = abs(voltage_V) / f
    th_star = 0.0
    for _ in range(12):
        v, th_star = _crest(prof, scale, frequency_Hz, t_in_s, ke_in_eV, mass_eV, charge)
        if v <= 0 or not math.isfinite(v):
            raise ValueError("no accelerating phase found for the RF profile")
        if abs(v - abs(voltage_V)) <= 1e-11 * abs(voltage_V):
            break
        scale *= abs(voltage_V) / v
    target = abs(voltage_V) * math.cos(phase_sync_rad)
    th = th_star + math.degrees(phase_sync_rad)
    if abs(math.cos(phase_sync_rad)) >= _CREST_COS:
        return scale, (th + 180.0) % 360.0 - 180.0          # near the crest the sinusoid's phase is the answer

    def gain(t):
        return _integrate(prof, scale, t, frequency_Hz, t_in_s, ke_in_eV, mass_eV, charge)[0] - ke_in_eV
    d = 1e-3
    for _ in range(30):
        g0 = gain(th) - target
        if abs(g0) <= 1e-15 * abs(voltage_V):           # the integrator's own roundoff (nano-eV on a MV)
            break
        slope = (gain(th + d) - gain(th - d)) / (2.0 * d)
        if slope == 0.0:
            break
        step = max(-30.0, min(30.0, g0 / slope))
        if abs(step) < 1e-13:
            break
        th -= step
    return scale, (th + 180.0) % 360.0 - 180.0
