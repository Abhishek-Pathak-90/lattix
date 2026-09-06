"""OPAL's one-dimensional field-map files (``Fieldmap.cpp``, ``FM1DDynamic.cpp``, ``FM1DMagnetoStatic.cpp``
of OPAL-X 078ff1e): a magic first word, a ``zbegin zend n`` line in **cm** (``n`` intervals: OPAL reads
``n + 1`` values), the frequency in MHz for a dynamic map, a radial line ``rbegin rend nr`` (read, unused
on axis), then one on-axis value per line.  OPAL normalises a map to its largest absolute value unless the
first line ends with ``FALSE``, expands it in ``accuracy`` Fourier terms (never more than the number of
points) and sets the element's length to the map's extent."""
from __future__ import annotations

import math
from dataclasses import dataclass

DYNAMIC = "1DDynamic"
STATIC = "1DMagnetoStatic"


@dataclass
class MapData:
    kind: str                       # DYNAMIC | STATIC
    z_m: list[float]                # uniform grid, ``zbegin`` … ``zend``
    values: list[float]             # E_z [V/m] or B_z [T] as written (OPAL normalises them itself)
    frequency_Hz: float | None = None
    normalized: bool = True

    @property
    def length_m(self) -> float:
        return self.z_m[-1] - self.z_m[0] if len(self.z_m) > 1 else 0.0

    @property
    def peak(self) -> float:
        return max((abs(v) for v in self.values), default=0.0)


def _num(v: float) -> str:
    s = f"{float(v):.12g}"
    return s if ("e" in s or "." in s or "n" in s) else s + ".0"


def _uniform(z_m: list[float], values: list[float], n: int | None) -> tuple[list[float], list[float]]:
    """Resample onto a uniform grid (OPAL assumes equal spacing: it reads values only)."""
    import numpy as np

    z = np.asarray(z_m, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(z) < 2:
        raise ValueError("a field map needs at least two points")
    n = int(n or len(z))
    n = max(2, n)
    zz = np.linspace(z[0], z[-1], n)
    return zz.tolist(), np.interp(zz, z, v).tolist()


def dynamic_map_text(z_m: list[float], ez_V_per_m: list[float], frequency_Hz: float, *, n: int | None = None,
                     accuracy: int | None = None, r_max_m: float = 0.0) -> str:
    """A ``1DDynamic`` file: OPAL scales the (normalised) profile by the cavity's ``VOLT`` [MV/m]."""
    z, v = _uniform(z_m, ez_V_per_m, n)
    acc = int(accuracy or len(z))
    lines = [f"{DYNAMIC} {acc}",
             f"{_num(100.0 * z[0])} {_num(100.0 * z[-1])} {len(z) - 1}",
             _num(frequency_Hz * 1e-6),
             f"0.0 {_num(100.0 * r_max_m)} 0"]
    lines += [_num(x) for x in v]
    return "\n".join(lines) + "\n"


def static_map_text(z_m: list[float], bz_T: list[float], *, n: int | None = None, accuracy: int | None = None,
                    r_max_m: float = 0.0) -> str:
    """A ``1DMagnetoStatic`` file: OPAL scales the (normalised) profile by ``KS · P0/c`` [T]."""
    z, v = _uniform(z_m, bz_T, n)
    acc = int(accuracy or len(z))
    lines = [f"{STATIC} {acc}",
             f"{_num(100.0 * z[0])} {_num(100.0 * z[-1])} {len(z) - 1}",
             f"0.0 {_num(100.0 * r_max_m)} 0"]
    lines += [_num(x) for x in v]
    return "\n".join(lines) + "\n"


def parse_map(text: str) -> MapData:
    """Read a ``1DDynamic`` / ``1DMagnetoStatic`` file the way ``readFileHeader`` + ``readFileData`` do."""
    rows = [ln.split("#")[0].strip() for ln in text.splitlines()]
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("empty field-map file")
    head = rows[0].split()
    magic = head[0]
    if magic.lower().startswith("1ddy"):
        kind = DYNAMIC
    elif magic.lower().startswith("1dma"):
        kind = STATIC
    else:
        raise ValueError(f"not a 1-D OPAL field map (first word {magic!r}); lattix reads 1DDynamic and "
                         "1DMagnetoStatic maps only")
    normalized = not (len(head) >= 3 and head[2].upper() == "FALSE")
    z0, z1, n_int = rows[1].split()[:3]
    z0, z1, n_int = float(z0) * 1e-2, float(z1) * 1e-2, int(float(n_int))
    i = 2
    freq = None
    if kind == DYNAMIC:
        freq = float(rows[i].split()[0]) * 1e6
        i += 1
    i += 1                                   # the radial line
    vals = [float(r.split()[0]) for r in rows[i:i + n_int + 1]]
    if len(vals) != n_int + 1:
        raise ValueError(f"field map declares {n_int + 1} values, found {len(vals)}")
    z = [z0 + (z1 - z0) * k / n_int for k in range(n_int + 1)] if n_int else [z0]
    return MapData(kind=kind, z_m=z, values=vals, frequency_Hz=freq, normalized=normalized)


def raised_bump(length_m: float, active_m: float, n: int) -> tuple[list[float], list[float]]:
    """``1 − cos`` accelerating bump of ``active`` width centred in ``length``, ``z`` from 0 to ``length``."""
    from lattix.formats.impactt.rfprofile import raised_cosine

    z, ez = raised_cosine(length_m, active_m, n)
    return [zz + 0.5 * length_m for zz in z], ez


def cell_profile(length_m: float, n_cell: int, *, pi_mode: bool, n: int | None = None,
                 active_m: float | None = None) -> tuple[list[float], list[float]]:
    from lattix.formats.impactt.rfprofile import cell_train

    z, ez = cell_train(length_m, n_cell, pi_mode=pi_mode, n=n, active=active_m)
    return [zz + 0.5 * length_m for zz in z], ez


def plateau(length_m: float, ramp_m: float, n: int) -> tuple[list[float], list[float]]:
    """A flat top of unit height with raised-cosine ramps of width ``ramp`` at both ends, spanning
    ``length + ramp`` so that ``∫ dz = length`` exactly (the hard-edge element of that length)."""
    n = max(21, int(n) | 1)
    total = length_m + ramp_m
    z = [total * i / (n - 1) for i in range(n)]
    out = []
    for zz in z:
        if zz < ramp_m:
            out.append(0.5 * (1.0 - math.cos(math.pi * zz / ramp_m)) if ramp_m > 0 else 1.0)
        elif zz > total - ramp_m:
            out.append(0.5 * (1.0 - math.cos(math.pi * (total - zz) / ramp_m)) if ramp_m > 0 else 1.0)
        else:
            out.append(1.0)
    return z, out
