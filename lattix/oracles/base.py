"""Oracle harness core: one interface, N engine adapters, one common basis.

An *oracle* runs a real accelerator code on a deck file and returns linear
optics in a uniform :class:`OracleResult`.  Adapters declare the coordinate
basis their engine uses; :meth:`OracleResult.to_common` maps everything onto
the common basis ``C = (x [m], px/p0, y [m], py/p0, z [m] ahead-positive, δ)``
using the *local* (β, γ, p0) at each element boundary (see ``basis.py``).

Nothing here depends on the lattix IR: Phase 0 validates engines against
each other on existing deck files before any translator code exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

C_LIGHT = 299_792_458.0  # m/s

SPECIES: dict[str, tuple[float, int]] = {
    # name: (mass [eV/c^2], charge [e])   -- H- mass = m_p + 2 m_e (HELIX particle.py:24)
    "proton": (938_272_088.16, +1),
    "h-": (938_272_088.16 + 2 * 510_998.95, -1),
    "electron": (510_998.95, -1),
    "positron": (510_998.95, +1),
    "deuteron": (1_875_612_942.6, +1),
}


class Basis(StrEnum):
    """Coordinate basis an engine reports its matrices in (see basis.py)."""

    COMMON = "common"      # (x, px/p0, y, py/p0, z ahead-positive [m], delta)
    MADX = "madx"          # (x, px, y, py, t, pt)
    XTRACK = "xtrack"      # (x, px, y, py, zeta, delta)
    BMAD = "bmad"          # (x, px, y, py, z, pz)
    ELEGANT = "elegant"    # (x, x', y, y', s, delta)
    TRACEWIN = "tracewin"  # (x, x', y, y', z, dp/p)  -- Transfer_matrix1.dat
    HELIX = "helix"        # (x [mm], x' [mrad], y [mm], y' [mrad], dphi [deg], dW [MeV])
    FLAME = "flame"        # (x [mm], x' [rad], y [mm], y' [rad], phi [rad], dEk [MeV/u])
    IMPACTX = "impactx"    # (x, px, y, py, t, pt)
    IMPACTZ = "impactz"    # (x, px/mc, y, py/mc, dphi, dE/mc^2)


@dataclass
class BeamSpec:
    """What the engine needs to know about the reference particle and the
    initial optics (open lines are propagated from these Twiss values)."""

    species: str = "proton"
    kinetic_energy_eV: float = 2.1e6
    frequency_Hz: float | None = None
    betx: float = 10.0
    alfx: float = 0.0
    bety: float = 10.0
    alfy: float = 0.0
    dx: float = 0.0
    dpx: float = 0.0
    dy: float = 0.0
    dpy: float = 0.0

    @property
    def mass_eV(self) -> float:
        return SPECIES[self.species.lower()][0]

    @property
    def charge(self) -> int:
        return SPECIES[self.species.lower()][1]

    @property
    def gamma(self) -> float:
        return 1.0 + self.kinetic_energy_eV / self.mass_eV

    @property
    def beta(self) -> float:
        g = self.gamma
        return float(np.sqrt(1.0 - 1.0 / (g * g)))

    @property
    def pc_eV(self) -> float:
        return self.beta * self.gamma * self.mass_eV

    @property
    def total_energy_eV(self) -> float:
        return self.gamma * self.mass_eV

    @property
    def brho(self) -> float:
        """Unsigned magnetic rigidity [T·m]."""
        return self.pc_eV / (C_LIGHT * abs(self.charge))


@dataclass
class Probe:
    """A small bunch, in the common basis, tracked through every engine."""

    coords: np.ndarray  # (N, 6)

    @classmethod
    def default(cls, n: int = 64, seed: int = 20260903) -> Probe:
        rng = np.random.default_rng(seed)
        amp = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-4])
        return cls(coords=rng.standard_normal((n, 6)) * amp)


@dataclass
class OracleResult:
    """Per-element linear optics from one engine, in that engine's basis."""

    engine: str
    basis: Basis
    names: list[str]
    length: np.ndarray               # (N,) element lengths [m]
    s_out: np.ndarray                # (N,) exit positions [m]
    R_elem: np.ndarray               # (N, 6, 6) entry->exit maps, native basis
    ref_kinetic_eV_in: np.ndarray    # (N,) reference kinetic energy at entry [eV]
    ref_kinetic_eV_out: np.ndarray   # (N,) ... at exit
    mass_eV: float
    charge: int
    rf_frequency_Hz: np.ndarray | None = None   # (N,) machine RF clock per element (phase bases)
    twiss: dict[str, np.ndarray] = field(default_factory=dict)   # betx alfx bety alfy at exits
    disp: dict[str, np.ndarray] = field(default_factory=dict)    # dx dpx dy dpy at exits (native)
    survey: np.ndarray | None = None                             # (N, 4) X, Y, Z, theta at exits
    probe_out: np.ndarray | None = None                          # (Np, 6) native basis
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- derived -------------------------------------------------------------
    def __post_init__(self) -> None:
        n = len(self.names)
        for attr in ("length", "s_out", "ref_kinetic_eV_in", "ref_kinetic_eV_out"):
            v = np.asarray(getattr(self, attr), dtype=float)
            if v.shape != (n,):
                raise ValueError(f"{attr} must have shape ({n},), got {v.shape}")
            setattr(self, attr, v)
        self.R_elem = np.asarray(self.R_elem, dtype=float)
        if self.R_elem.shape != (n, 6, 6):
            raise ValueError(f"R_elem must have shape ({n},6,6), got {self.R_elem.shape}")

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def R_cum(self) -> np.ndarray:
        """(N, 6, 6) start->exit cumulative maps."""
        out = np.empty_like(self.R_elem)
        acc = np.eye(6)
        for i in range(self.n):
            acc = self.R_elem[i] @ acc
            out[i] = acc
        return out

    @property
    def total_length(self) -> float:
        return float(self.length.sum())

    def gamma_in(self) -> np.ndarray:
        return 1.0 + self.ref_kinetic_eV_in / self.mass_eV

    def gamma_out(self) -> np.ndarray:
        return 1.0 + self.ref_kinetic_eV_out / self.mass_eV

    def to_common(self) -> OracleResult:
        """Return a copy with ``R_elem`` (and probe/dispersion) in the common basis."""
        from lattix.oracles.basis import transform_matrix

        if self.basis is Basis.COMMON:
            return self
        R = np.empty_like(self.R_elem)
        f = self.rf_frequency_Hz
        for i in range(self.n):
            t_in = transform_matrix(self.basis, self.ref_kinetic_eV_in[i], self.mass_eV,
                                    None if f is None else float(f[i]))
            t_out = transform_matrix(self.basis, self.ref_kinetic_eV_out[i], self.mass_eV,
                                     None if f is None else float(f[i]))
            R[i] = t_out @ self.R_elem[i] @ np.linalg.inv(t_in)
        probe = None
        if self.probe_out is not None:
            t_end = transform_matrix(self.basis, self.ref_kinetic_eV_out[-1], self.mass_eV,
                                     None if f is None else float(f[-1]))
            probe = self.probe_out @ t_end.T
        return OracleResult(
            engine=self.engine, basis=Basis.COMMON, names=list(self.names),
            length=self.length.copy(), s_out=self.s_out.copy(), R_elem=R,
            ref_kinetic_eV_in=self.ref_kinetic_eV_in.copy(),
            ref_kinetic_eV_out=self.ref_kinetic_eV_out.copy(),
            mass_eV=self.mass_eV, charge=self.charge, rf_frequency_Hz=self.rf_frequency_Hz,
            twiss=dict(self.twiss), disp=dict(self.disp), survey=self.survey,
            probe_out=probe, warnings=list(self.warnings), meta=dict(self.meta),
        )


@runtime_checkable
class Oracle(Protocol):
    """Engine adapter contract.  ``formats`` lists the deck formats the engine
    can consume directly (e.g. ``("madx",)``)."""

    name: str
    formats: tuple[str, ...]

    def available(self) -> tuple[bool, str]: ...

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None) -> OracleResult: ...


_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: make an adapter discoverable by ``get_oracle(name)``."""
    _REGISTRY[cls.name] = cls
    return cls


def _load_adapters() -> None:
    import importlib

    for mod in ("cpymad", "xtrack", "pytao", "helix", "elegant", "tracewin", "impactx", "impactz", "flame",
                "scibmad"):
        try:
            importlib.import_module(f"lattix.oracles.{mod}")
        except ModuleNotFoundError as e:  # adapter module itself missing (not its engine)
            if not str(e).startswith("No module named 'lattix.oracles"):
                raise


def get_oracle(name: str) -> Oracle:
    if name not in _REGISTRY:
        _load_adapters()
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise KeyError(f"unknown oracle {name!r}; known: {sorted(_REGISTRY)}") from None


def available_oracles() -> dict[str, tuple[bool, str]]:
    _load_adapters()
    return {n: cls().available() for n, cls in sorted(_REGISTRY.items())}


def guess_format(path: Path) -> str:
    """Deck format from suffix (the single registry rule, see PLAN §2)."""
    s = path.suffix.lower()
    return {
        ".madx": "madx", ".seq": "madx", ".mad": "madx", ".str": "madx",
        ".bmad": "bmad", ".lte": "elegant", ".lat": "mad8", ".flat": "mad8",
        ".dat": "tracewin", ".yaml": "pals", ".yml": "pals", ".json": "xtrack",
    }.get(s, s.lstrip("."))
