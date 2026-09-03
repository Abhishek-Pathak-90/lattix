"""Reference particle: species, kinetic energy, RF clock.

Rigidity policy (PLAN §4.1): ``brho_abs = pc/(|q|c)``; ``brho_signed = sign(q)·brho_abs``.
MAD-family normalized strengths (K1, ks, KnL, kicks) always use ``brho_signed`` so an
H⁻ deck flips consistently (HELIX ``madx_parser._signed_brho``).
"""
from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from lattix.ir.units import C_LIGHT


class Species(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    mass_eV: float
    charge: int


_M_E = 510_998.95
_M_P = 938_272_088.16
SPECIES: dict[str, Species] = {
    "proton": Species(name="proton", mass_eV=_M_P, charge=+1),
    "antiproton": Species(name="antiproton", mass_eV=_M_P, charge=-1),
    "h-": Species(name="h-", mass_eV=_M_P + 2 * _M_E, charge=-1),   # HELIX particle.py:24
    "electron": Species(name="electron", mass_eV=_M_E, charge=-1),
    "positron": Species(name="positron", mass_eV=_M_E, charge=+1),
    "deuteron": Species(name="deuteron", mass_eV=1_875_612_942.6, charge=+1),
}
_ALIASES = {"p": "proton", "hminus": "h-", "h_minus": "h-", "e-": "electron", "e": "electron",
            "e+": "positron", "d": "deuteron", "pbar": "antiproton"}


def species(name: str | Species) -> Species:
    if isinstance(name, Species):
        return name
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in SPECIES:
        raise KeyError(f"unknown species {name!r}; known: {sorted(SPECIES)}")
    return SPECIES[key]


class ReferenceParticle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    species: Species
    kinetic_energy_eV: float
    rf_frequency_Hz: float | None = None    # machine RF clock at this point (TraceWin FREQ state)
    time_s: float = 0.0

    @property
    def gamma(self) -> float:
        return 1.0 + self.kinetic_energy_eV / self.species.mass_eV

    @property
    def beta(self) -> float:
        g = self.gamma
        return math.sqrt(1.0 - 1.0 / (g * g))

    @property
    def total_energy_eV(self) -> float:
        return self.gamma * self.species.mass_eV

    @property
    def pc_eV(self) -> float:
        return self.beta * self.gamma * self.species.mass_eV

    @property
    def brho_abs(self) -> float:
        return self.pc_eV / (C_LIGHT * abs(self.species.charge))

    @property
    def brho_signed(self) -> float:
        return math.copysign(self.brho_abs, self.species.charge)

    @property
    def wavelength_m(self) -> float | None:
        return None if not self.rf_frequency_Hz else C_LIGHT / self.rf_frequency_Hz

    def advanced(self, *, dE_eV: float = 0.0, ds_m: float = 0.0,
                 rf_frequency_Hz: float | None = None) -> ReferenceParticle:
        """Copy with energy gain, path length (time of flight) and/or a new RF clock."""
        dt = ds_m / (self.beta * C_LIGHT) if ds_m else 0.0
        return self.model_copy(update={
            "kinetic_energy_eV": self.kinetic_energy_eV + dE_eV,
            "time_s": self.time_s + dt,
            "rf_frequency_Hz": self.rf_frequency_Hz if rf_frequency_Hz is None else rf_frequency_Hz,
        })

    @classmethod
    def from_total_energy(cls, sp: str | Species, total_energy_eV: float, **kw) -> ReferenceParticle:
        s = species(sp)
        return cls(species=s, kinetic_energy_eV=total_energy_eV - s.mass_eV, **kw)

    @classmethod
    def from_momentum(cls, sp: str | Species, pc_eV: float, **kw) -> ReferenceParticle:
        s = species(sp)
        return cls(species=s, kinetic_energy_eV=math.hypot(pc_eV, s.mass_eV) - s.mass_eV, **kw)

    @classmethod
    def from_brho(cls, sp: str | Species, brho: float, **kw) -> ReferenceParticle:
        s = species(sp)
        return cls.from_momentum(s, abs(brho) * C_LIGHT * abs(s.charge), **kw)
