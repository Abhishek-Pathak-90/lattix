"""RF phase conventions per format (PLAN §4.3; signs measured 2026-09-03).

IR: synchronous phase φ [rad], 0 = crest, gain = V·cos φ for the reference particle.
"""
from __future__ import annotations

import math

from lattix.ir.units import rad_to_turns, turns_to_rad, wrap_rad


def energy_gain_eV(voltage_V: float, phase_rad: float) -> float:
    return voltage_V * math.cos(phase_rad)


# MAD-X / xtrack: gain = V·sin(2π·lag), species-independent (cpymad: charge ±1 same gain)
def madx_lag(phase_rad: float) -> float:
    return rad_to_turns(phase_rad) + 0.25


def phase_from_madx_lag(lag: float) -> float:
    return wrap_rad(turns_to_rad(lag - 0.25))


# Bmad lcavity: phi0 in turns, 0 = crest, species-independent
def bmad_phi0(phase_rad: float) -> float:
    return rad_to_turns(phase_rad)


def phase_from_bmad_phi0(phi0: float) -> float:
    return wrap_rad(turns_to_rad(phi0))


# Elegant RFCA: phase in degrees, crest at +90° for negative species, −90° for positive
def elegant_phase_deg(phase_rad: float, charge: int) -> float:
    crest = 90.0 if charge < 0 else -90.0
    return (math.degrees(phase_rad) + crest) % 360.0


def phase_from_elegant_deg(phase_deg: float, charge: int) -> float:
    crest = 90.0 if charge < 0 else -90.0
    return wrap_rad(math.radians(phase_deg - crest))


# TraceWin / HELIX: GAP/FIELD_MAP phase in degrees; the deck phase is a synchronous phase
# after SET_SYNC_PHASE (species-independent) and a raw RF phase otherwise, for which
# gain = q·E0TL·cos(φ_rf): negative species need a π shift to the IR's synchronous phase.
def tracewin_phase_deg(phase_rad: float, charge: int, sync: bool) -> float:
    shift = 0.0 if (sync or charge > 0) else math.pi
    return math.degrees(wrap_rad(phase_rad + shift))


def phase_from_tracewin_deg(phase_deg: float, charge: int, sync: bool) -> float:
    shift = 0.0 if (sync or charge > 0) else math.pi
    return wrap_rad(math.radians(phase_deg) + shift)


# PALS RFP.phase in turns, zero_phase = ACCELERATING → 0 = crest
def pals_phase(phase_rad: float) -> float:
    return rad_to_turns(phase_rad)


# TraceWin/HELIX thin gap: after the energy kick and the adiabatic damping, a round thin lens
#     Δx' = k·x, Δy' = k·y,   k = −π·V_eff·sin φ / (m c² · (βγ)_out³ · λ)      [1/m]
# with φ the IR synchronous phase (species-independent: q·sin φ_rf = sin φ for either charge
# sign), (βγ)_out the reference particle after the gap and λ the gap's RF wavelength
# (HELIX linac_gen/elements/rf_gap.py::kick_matrix, lockstep with TraceWin to 1e-6).
def thin_gap_defocusing(voltage_V: float, phase_rad: float, frequency_Hz: float | None,
                        mass_eV: float, betagamma_out: float) -> float:
    """Transverse RF defocusing strength k [1/m] of a thin gap (0 when the gap has no field
    or no frequency)."""
    if not voltage_V or not frequency_Hz or betagamma_out <= 0:
        return 0.0
    wavelength = 299792458.0 / float(frequency_Hz)
    return -math.pi * voltage_V * math.sin(phase_rad) / (mass_eV * betagamma_out ** 3 * wavelength)
