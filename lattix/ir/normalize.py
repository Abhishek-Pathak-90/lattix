"""Lab-field ⇄ normalized strength conversions (all through the SIGNED rigidity)."""
from __future__ import annotations

from lattix.ir.reference import ReferenceParticle


def k1_from_gradient(G_T_per_m: float, ref: ReferenceParticle) -> float:
    return G_T_per_m / ref.brho_signed


def gradient_from_k1(k1: float, ref: ReferenceParticle) -> float:
    return k1 * ref.brho_signed


def kn_from_bn(order: int, Bn: float, ref: ReferenceParticle) -> float:
    """Normalized K_n = B_n / Bρ for any order (per-length or integrated alike)."""
    return Bn / ref.brho_signed


def bn_from_kn(order: int, Kn: float, ref: ReferenceParticle) -> float:
    return Kn * ref.brho_signed


def ks_from_field(Bsol_T: float, ref: ReferenceParticle) -> float:
    return Bsol_T / ref.brho_signed


def field_from_ks(ks: float, ref: ReferenceParticle) -> float:
    return ks * ref.brho_signed


def field_index_from_k1(k1: float, rho_m: float) -> float:
    """TraceWin/HELIX dipole field index N = −k1·ρ² (HELIX madx_parser)."""
    return -k1 * rho_m * rho_m


def k1_from_field_index(N: float, rho_m: float) -> float:
    return -N / (rho_m * rho_m)


def kick_from_bl(bl_Tm: float, ref: ReferenceParticle) -> float:
    """Deflection [rad] of the reference particle from an integrated field ∫B·dl [T·m]."""
    return bl_Tm / ref.brho_signed


def bl_from_kick(kick_rad: float, ref: ReferenceParticle) -> float:
    return kick_rad * ref.brho_signed
