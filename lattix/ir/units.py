"""Unit conversions.  The IR is SI + eV: m, rad, T, T/m, V, V/m, Hz, s, eV, eV/c."""
from __future__ import annotations

import math

C_LIGHT = 299_792_458.0          # m/s
E_CHARGE = 1.602_176_634e-19     # C

MM = 1e-3
DEG = math.pi / 180.0
MEV = 1e6
GEV = 1e9
MHZ = 1e6
TURN = 2.0 * math.pi             # rad per turn (MAD-X lag, Bmad phi0, PALS phase are in turns)


def mm_to_m(x: float) -> float:
    return x * MM


def m_to_mm(x: float) -> float:
    return x / MM


def deg_to_rad(x: float) -> float:
    return x * DEG


def rad_to_deg(x: float) -> float:
    return x / DEG


def turns_to_rad(x: float) -> float:
    return x * TURN


def rad_to_turns(x: float) -> float:
    return x / TURN


def wrap_rad(x: float) -> float:
    """Wrap an angle to (−π, π]."""
    y = math.fmod(x + math.pi, TURN)
    if y <= 0:
        y += TURN
    return y - math.pi
