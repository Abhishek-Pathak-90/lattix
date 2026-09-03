"""TraceWin field-map data, on-axis profiles and the reference-energy integral (PLAN §4.3).

Two things live here:

* :class:`FieldMapData` — the *data* behind a :class:`~lattix.ir.elements.FieldMap` card:
  one :class:`FieldChannelData` per enabled ``geom`` channel (``STAT_E``, ``STAT_B``,
  ``RF_E``, ``RF_B``), each with its grid axes, its raw components and the on-axis
  ``Fz(z)`` profile.  The file layouts and loop orders are ports of HELIX
  ``linac_gen/io/field_map_reader.py`` and ``linac_gen/io/tracewin_fieldmap_reader.py``
  (1-D, 2-D cylindrical, 2-D Cartesian, 3-D Cartesian, plus the legacy Linac_Gen 1-D
  layout), **converted to SI on the way in**: TraceWin writes metres and MV/m (E) or T
  (B), lattix stores m, V/m, T, T/m.
* :func:`integrate_map` — the synchronous-phase integration that gives a map its
  reference energy gain, its transit-time factor and its solenoid/quadrupole equivalents.
  It is a port of HELIX ``linac_gen/elements/field_map.py`` (``advance_ref``,
  ``_probe_voltage_integral``, ``_calibrate_sync_phase``, ``_phi_sync_rad``) — same
  midpoint quadrature, same step count, same ``SET_SYNC_PHASE`` fixed-point iteration —
  so lattix and HELIX agree on ``dE_ref`` to round-off on the real PIP-II maps.

The integration scheme (HELIX ``advance_ref``), for ``n`` sub-steps of length
``ds = L_card/n`` over the card's length, is::

    z_i   = z_map_start + (i + ½)·ds
    φ_i   = φ_entrance + Σ_{k<i} 360·ds/(β_k·λ) + 180·ds/(β_i·λ)     [deg]
    dW_i  = Σ_channels q·k·F_z(z_i)/Norm · phasor(φ_i) · ds          [eV]
    W    += dW_i,  then β is recomputed from the *new* W

with ``phasor = 1`` for a static channel and ``cos φ`` for an RF electric one (a magnetic
field does no work on the reference).  ``n = max(N_z − 1, 50)`` where ``N_z`` is the number
of z samples of the first enabled channel — HELIX's rule, so the quadrature nodes fall on
the map's own grid.

``SET_SYNC_PHASE`` (``p_flag = 1``) makes the deck's θ a *synchronous* phase: the RF phase
actually applied is ``θ − ψ`` where ψ is the fixed point of ``ψ = arg V_c(θ − ψ)`` and
``V_c`` is the complex voltage integrated with the β evolution the run itself produces
(HELIX ``_calibrate_sync_phase``: 30 iterations, 0.7 under-relaxation, 0.005° tolerance).
At the fixed point ``dE_ref = |V_c|·cos θ``, which is exactly the IR's own convention
(``gain = V_eff·cos φ``, 0 = crest, species-independent) — so an H⁻ deck and a proton deck
with the same θ gain the same energy, and ``phase_sync_rad`` comes back as θ.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lattix.ir.reference import ReferenceParticle
from lattix.ir.rf import tracewin_phase_deg
from lattix.ir.units import C_LIGHT

# ---------------------------------------------------------------------------
# Channels.  Strings, not the ``formats.tracewin`` enum, so ``lattix.ir`` keeps no
# import-time dependency on a format package (the loader imports the geom decoder lazily).
STAT_E = "STAT_E"
STAT_B = "STAT_B"
RF_E = "RF_E"
RF_B = "RF_B"
#: fixed iteration order, matching ``fieldmap_files.enabled_channels``
CHANNEL_ORDER: tuple[str, ...] = (STAT_E, STAT_B, RF_E, RF_B)


def channel_is_electric(channel: str) -> bool:
    return channel.endswith("_E")


def channel_is_static(channel: str) -> bool:
    return channel.startswith("STAT")


#: value of ``ds`` below which a map is treated as having no length
_L_TOL = 1e-15


# ---------------------------------------------------------------------------
# Per-component file readers (SI on the way out).  HELIX ports; see module docstring.
def _clean_lines(path: str | os.PathLike) -> list[str]:
    with open(path, encoding="latin-1") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith(("#", "!"))]


def _flat(lines: list[str], count: int, path: str) -> np.ndarray:
    try:
        return np.fromiter((float(v) for ln in lines for v in ln.split()), dtype=float, count=count)
    except ValueError as exc:  # too few values, or a non-numeric token
        raise ValueError(f"{path}: expected {count} field values ({exc})") from exc


def _si_factor(suffix: str) -> float:
    """``.e**`` files hold MV/m, ``.b**`` files hold T (or T/m for a ``geom`` digit 9 map)."""
    return 1e6 if suffix[1:2].lower() == "e" else 1.0


@dataclass(frozen=True)
class _Raw:
    """One component file: axes in m, values in SI, plus the header's ``Norm``."""

    values: np.ndarray
    norm: float
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    r: np.ndarray | None = None


def _read_1d(path: str, unit: float) -> _Raw:
    """1-D ``F_z(z)``.  Canonical TraceWin: ``Nz Zmax[m] / Norm / (Nz+1)×F``.

    The legacy Linac_Gen layout (``N_pts / zmin[cm] zmax[cm] / N_pts×F``) is auto-detected
    exactly as HELIX ``read_1d_component`` does, so old fixtures keep working.
    """
    raw = _clean_lines(path)
    tok = raw[0].split()
    if len(tok) >= 2:
        try:
            nz = int(tok[0])
            float(tok[1])
            if "." in tok[1] or "e" in tok[1].lower():
                zmax = float(tok[1])
                norm = float(raw[1].split()[0])
                vals = _flat(raw[2:], nz + 1, path) * unit
                return _Raw(values=vals, norm=norm, z=np.linspace(0.0, zmax, nz + 1))
        except ValueError:
            pass
    n_pts = int(tok[0])
    z0_cm, z1_cm = (float(v) for v in raw[1].split()[:2])
    vals = _flat(raw[2:], n_pts, path) * unit
    return _Raw(values=vals, norm=1.0, z=np.linspace(z0_cm * 1e-2, z1_cm * 1e-2, n_pts))


def _read_2d_cyl(path: str, unit: float) -> _Raw:
    """2-D cylindrical: ``Nz Zmax / Nr Rmax / Norm / for k∈z: for i∈r`` (r fastest).

    Returned with shape ``(Nz+1, Nr+1)`` — column 0 is the on-axis (r = 0) slice.
    """
    raw = _clean_lines(path)
    t0, t1 = raw[0].split(), raw[1].split()
    nz, zmax = int(t0[0]), float(t0[1])
    nr, rmax = int(t1[0]), float(t1[1])
    norm = float(raw[2].split()[0])
    vals = _flat(raw[3:], (nz + 1) * (nr + 1), path) * unit
    return _Raw(values=vals.reshape((nz + 1, nr + 1)), norm=norm,
                z=np.linspace(0.0, zmax, nz + 1), r=np.linspace(0.0, rmax, nr + 1))


def _read_2d_cart(path: str, unit: float) -> _Raw:
    """2-D Cartesian: ``Nx Xmin Xmax / Ny Ymin Ymax / Norm / for j∈y: for i∈x`` (x fastest).

    Returned with shape ``(Nx+1, Ny+1)``; the map is invariant along z (no z axis).
    """
    raw = _clean_lines(path)
    tx, ty = raw[0].split(), raw[1].split()
    nx, xmin, xmax = int(tx[0]), float(tx[1]), float(tx[2])
    ny, ymin, ymax = int(ty[0]), float(ty[1]), float(ty[2])
    norm = float(raw[2].split()[0])
    vals = _flat(raw[3:], (nx + 1) * (ny + 1), path) * unit
    return _Raw(values=vals.reshape((ny + 1, nx + 1)).T.copy(), norm=norm,
                x=np.linspace(xmin, xmax, nx + 1), y=np.linspace(ymin, ymax, ny + 1))


def _read_3d_cart(path: str, unit: float) -> _Raw:
    """3-D Cartesian: ``Nz Zmax / Nx Xmin Xmax / Ny Ymin Ymax / Norm`` then
    ``for k∈z: for j∈y: for i∈x`` (x fastest).  Returned with shape ``(Nx+1, Ny+1, Nz+1)``."""
    raw = _clean_lines(path)
    tz, tx, ty = raw[0].split(), raw[1].split(), raw[2].split()
    nz, zmax = int(tz[0]), float(tz[1])
    nx, xmin, xmax = int(tx[0]), float(tx[1]), float(tx[2])
    ny, ymin, ymax = int(ty[0]), float(ty[1]), float(ty[2])
    norm = float(raw[3].split()[0])
    vals = _flat(raw[4:], (nx + 1) * (ny + 1) * (nz + 1), path) * unit
    arr = vals.reshape((nz + 1, ny + 1, nx + 1)).transpose(2, 1, 0).copy()
    return _Raw(values=arr, norm=norm, x=np.linspace(xmin, xmax, nx + 1),
                y=np.linspace(ymin, ymax, ny + 1), z=np.linspace(0.0, zmax, nz + 1))


_COMPONENT_READERS = {1: _read_1d, 9: _read_1d, 4: _read_2d_cyl, 5: _read_2d_cyl,
                      6: _read_2d_cart, 7: _read_3d_cart}

#: per-file cache keyed on (realpath, mtime, size) — a linac references the same map
#: template from dozens of cards (HELIX ``_cached_file_reader``; the 3-D PIP-II templates
#: are 2.4 MB each and appear 8× in ``mebt+hwr.dat``).
_FILE_CACHE: dict[tuple[str, float, int], _Raw] = {}
#: assembled maps, keyed on (geom, prefix) and validated against the files' (mtime, size)
_MAP_CACHE: dict[tuple[int, str], tuple[tuple, FieldMapData]] = {}


def read_component(path: str | os.PathLike, digit: int) -> _Raw:
    """Read one component file for a ``geom`` digit; cached on (path, mtime, size)."""
    p = str(path)
    reader = _COMPONENT_READERS.get(int(digit))
    if reader is None:
        raise ValueError(f"no field-map reader for geometry digit {digit}")
    try:
        st = os.stat(p)
        key = (os.path.realpath(p), st.st_mtime, st.st_size)
    except OSError:
        return reader(p, _si_factor(os.path.splitext(p)[1]))
    hit = _FILE_CACHE.get(key)
    if hit is None:
        hit = reader(p, _si_factor(os.path.splitext(p)[1]))
        for arr in (hit.values, hit.x, hit.y, hit.z, hit.r):
            if arr is not None:
                arr.flags.writeable = False          # a cached array is shared; freeze it
        _FILE_CACHE[key] = hit
    return hit


def _stamp(files: list[str]) -> tuple:
    """(mtime, size) of every component file — the freshness key of a cached map."""
    out = []
    for f in files:
        try:
            st = os.stat(f)
        except OSError:
            return ()
        out.append((st.st_mtime, st.st_size))
    return tuple(out)


def clear_cache() -> None:
    """Drop every cached component file and assembled map (tests that rewrite maps on disk)."""
    _FILE_CACHE.clear()
    _MAP_CACHE.clear()


def sha256_file(path: str | os.PathLike) -> str:
    """sha256 of one component file (cached) — invariant I-12."""
    from lattix.formats.tracewin.fieldmap_files import sha256

    return sha256(path)


# ---------------------------------------------------------------------------
@dataclass
class FieldChannelData:
    """One field channel of a map: grid axes in m, components in SI."""

    channel: str
    digit: int
    norm: float = 1.0
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    z: np.ndarray | None = None
    r: np.ndarray | None = None
    Fx: np.ndarray | None = None
    Fy: np.ndarray | None = None
    Fz: np.ndarray | None = None
    Fr: np.ndarray | None = None
    Fq: np.ndarray | None = None
    files: list[str] = field(default_factory=list)
    _axis: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False, compare=False)

    @property
    def is_electric(self) -> bool:
        return channel_is_electric(self.channel)

    @property
    def is_static(self) -> bool:
        return channel_is_static(self.channel)

    def on_axis(self) -> tuple[np.ndarray, np.ndarray] | None:
        """``(z [m], F_z on the axis)`` in SI — ``r = 0`` for a 2-D cylindrical map,
        ``x = y = 0`` for a 3-D Cartesian one.  ``None`` when the channel has no z axis
        (a 2-D Cartesian map is invariant along z) or no longitudinal component."""
        if self._axis is not None:
            return self._axis
        if self.z is None or self.Fz is None:
            return None
        f = np.asarray(self.Fz)
        if f.ndim == 1:
            line = f
        elif f.ndim == 2:                                 # (Nz+1, Nr+1), r = 0 is column 0
            if self.r is not None and f.shape == (len(self.z), len(self.r)):
                line = f[:, 0]
            else:
                line = f[0, :]
        else:
            # 3-D (Nx+1, Ny+1, Nz+1): bilinear at x = y = 0 (exact when 0 is a grid point)
            line = _interp_axis(self.y, _interp_axis(self.x, f, 0, at=0.0), 0, at=0.0)
        self._axis = (np.asarray(self.z), np.asarray(line))
        return self._axis

    def sample_on_axis(self, z_m: float) -> float:
        prof = self.on_axis()
        if prof is None:
            return 0.0
        zz, ff = prof
        return float(np.interp(z_m, zz, ff, left=0.0, right=0.0))


def _interp_axis(axis: np.ndarray | None, values: np.ndarray, axis_index: int = 0, *,
                 at: float = 0.0) -> np.ndarray:
    """Linear interpolation of *values* along *axis* at coordinate *at* (clamped)."""
    if axis is None or len(axis) == 1:
        return np.take(values, 0, axis=axis_index)
    a = np.asarray(axis, dtype=float)
    pos = float(np.clip(at, a[0], a[-1]))
    i = int(np.searchsorted(a, pos)) - 1
    i = max(0, min(i, len(a) - 2))
    t = 0.0 if a[i + 1] == a[i] else (pos - a[i]) / (a[i + 1] - a[i])
    lo = np.take(values, i, axis=axis_index)
    hi = np.take(values, i + 1, axis=axis_index)
    return lo * (1.0 - t) + hi * t


@dataclass
class FieldMapData:
    """Every channel a ``FIELD_MAP`` card refers to, loaded and in SI."""

    channels: dict[str, FieldChannelData] = field(default_factory=dict)
    geom: int | None = None
    prefix: str | None = None
    files: list[str] = field(default_factory=list)
    aperture_file: str | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, geom: int, map_dir: str | os.PathLike, base: str) -> FieldMapData:
        """Load every component file implied by ``geom`` off ``map_dir/base``.

        Raises ``FileNotFoundError`` naming every missing component, ``ValueError`` /
        ``NotImplementedError`` for a geometry digit lattix cannot read.
        """
        # local import: ``lattix.ir`` must not depend on a format package at import time
        from lattix.formats.tracewin.fieldmap_files import (
            component_files,
            decode_geom,
            enabled_channels,
        )

        code = decode_geom(geom)
        prefix = os.path.join(str(map_dir), base)
        key = (int(geom), os.path.realpath(prefix))
        hit = _MAP_CACHE.get(key)
        if hit is not None and hit[0] == _stamp(hit[1].files):
            return hit[1]
        out = cls(geom=int(geom), prefix=prefix)
        missing = []
        plan: list[tuple[str, int, list[str]]] = []
        for ch, digit in enabled_channels(code):
            sufs = component_files(ch, digit)
            plan.append((ch.name, digit, sufs))
            missing += [prefix + s for s in sufs if not Path(prefix + s).is_file()]
        if missing:
            raise FileNotFoundError("missing field-map file(s): " + ", ".join(missing))
        for name, digit, sufs in plan:
            data = FieldChannelData(channel=name, digit=digit)
            for suf in sufs:
                path = prefix + suf
                raw = read_component(path, digit)
                _attach(data, suf[-1], raw)
                if data.norm == 1.0 and raw.norm != 1.0:
                    data.norm = raw.norm
                data.files.append(path)
                out.files.append(path)
            out.channels[name] = data
        if code.aper != 0 and Path(prefix + ".ouv").is_file():
            out.aperture_file = prefix + ".ouv"
            out.files.append(out.aperture_file)
        _MAP_CACHE[key] = (_stamp(out.files), out)
        return out

    # -- queries ------------------------------------------------------------
    def channel(self, name: str) -> FieldChannelData | None:
        return self.channels.get(name)

    @property
    def ordered(self) -> list[FieldChannelData]:
        """Channels in the canonical STAT_E, STAT_B, RF_E, RF_B order."""
        return [self.channels[c] for c in CHANNEL_ORDER if c in self.channels]

    @property
    def z(self) -> np.ndarray:
        """The z axis of the first enabled channel that has one (m)."""
        for ch in self.ordered:
            if ch.z is not None:
                return np.asarray(ch.z)
        return np.zeros(0)

    @property
    def has_electric(self) -> bool:
        return any(ch.is_electric for ch in self.channels.values())

    def ez_profile(self, ke: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None:
        """On-axis ``E_z(z)`` [V/m] of the electric channels, scaled by ``ke/Norm``."""
        return self._profile([ch for ch in self.ordered if ch.is_electric and ch.digit != 9], ke)

    def bz_profile(self, kb: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None:
        """On-axis ``B_z(z)`` [T] of the magnetic channels (T/m for a digit-9 quad map)."""
        return self._profile([ch for ch in self.ordered if not ch.is_electric], kb)

    @staticmethod
    def _profile(chans: list[FieldChannelData], k: float) -> tuple[np.ndarray, np.ndarray] | None:
        parts = [(ch, ch.on_axis()) for ch in chans]
        parts = [(ch, p) for ch, p in parts if p is not None]
        if not parts:
            return None
        z = parts[0][1][0]
        total = np.zeros(len(z))
        for ch, (zz, ff) in parts:
            total += np.interp(z, zz, ff, left=0.0, right=0.0) * (k / ch.norm)
        return np.asarray(z), total


def _attach(data: FieldChannelData, letter: str, raw: _Raw) -> None:
    """Move a component file's values into the slot its suffix letter names."""
    for name in ("x", "y", "z", "r"):
        if getattr(raw, name) is not None and getattr(data, name) is None:
            setattr(data, name, getattr(raw, name))
    slot = {"z": "Fz", "r": "Fr", "x": "Fx", "y": "Fy", "q": "Fq"}.get(letter)
    if slot is None:
        raise ValueError(f"unrecognised field-map component letter {letter!r}")
    setattr(data, slot, raw.values)


# ---------------------------------------------------------------------------
# Reference-energy integration (HELIX ``advance_ref`` / ``_calibrate_sync_phase``)
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def _beta_of(w_kin_eV: float, mass_eV: float) -> float:
    """β from the kinetic energy, HELIX ``ReferenceParticle._update_derived`` (no 1−1/γ²
    cancellation at sub-MeV injection energies)."""
    gm1 = w_kin_eV / mass_eV
    gamma = 1.0 + gm1
    if gm1 <= 0.0:
        return 0.0
    return math.sqrt(gm1 * (gamma + 1.0)) / gamma


def _range_integral(z: np.ndarray, f: np.ndarray, z0: float, z1: float) -> float:
    """``∫ f dz`` over ``[z0, z1]`` by the trapezoid rule on the map's own grid, with the
    end points interpolated (so ∫B and ∫B² are the map's, not a re-sampling's)."""
    if z is None or len(z) < 2:
        return 0.0
    lo, hi = max(min(z0, z1), float(z[0])), min(max(z0, z1), float(z[-1]))
    if hi <= lo:
        return 0.0
    inner = z[(z > lo) & (z < hi)]
    zz = np.concatenate(([lo], inner, [hi]))
    return float(_TRAPZ(np.interp(zz, z, f), zz))


@dataclass(frozen=True)
class MapSummary:
    """What a field map does to the reference particle (PLAN §4.3).

    ``dE_ref_eV`` is HELIX's ``advance_ref`` number; ``v_c_V`` and ``phase_sync_rad`` are
    the *self-consistent* thin-gap equivalent (``dE_ref = v_c_V·cos(phase_sync_rad)``
    identically, β evolution folded in) that the writers emit.  ``v_eff_V`` / ``ttf`` are
    PLAN's fixed-β form: ``T = |∫E_z e^{iωz/β_in c} dz| / ∫|E_z| dz`` and
    ``V_eff = ∫|E_z| dz · T`` at the *entrance* β.
    """

    kind: str = "none"                    # rf | solenoid | quad | none
    length_m: float = 0.0
    n_steps: int = 0
    frequency_Hz: float | None = None
    beta_in: float = 0.0
    beta_out: float = 0.0
    # -- electric --------------------------------------------------------
    int_Ez_V: float = 0.0                 # ∫ k_e·E_z dz  (signed)
    int_abs_Ez_V: float = 0.0             # ∫ |k_e·E_z| dz
    ttf: float = 1.0
    v_eff_V: float = 0.0
    v_c_V: float = 0.0
    phase_rf_rad: float = 0.0             # RF phase applied at the map entrance (θ − ψ if sync)
    phase_sync_rad: float = 0.0           # synchronous phase: dE_ref = v_c_V·cos(phase_sync_rad)
    sync_offset_deg: float = 0.0          # ψ from the SET_SYNC_PHASE fixed point
    dE_ref_eV: float = 0.0
    dE_static_eV: float = 0.0             # part of dE_ref from a *static* electric channel
    phase_advance_deg: float = 0.0        # φ_s advance of the reference across the map
    # -- magnetic --------------------------------------------------------
    int_Bz_Tm: float = 0.0
    int_Bz2_T2m: float = 0.0
    L_eff_m: float = 0.0
    B_eff_T: float = 0.0
    int_Gz_Tm_per_m: float | None = None  # ∫G dz [T] for a geom digit 9 quadrupole map
    int_Gz2_T2m_per_m2: float | None = None  # ∫G² dz [T²/m]
    # -- provenance ------------------------------------------------------
    files: tuple[str, ...] = ()
    sha256: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        """JSON-able form for ``element.meta['map_summary']``."""
        d = {k: v for k, v in self.__dict__.items()}
        d["files"] = list(self.files)
        d["sha256"] = list(self.sha256)
        return d

    @property
    def is_accelerating(self) -> bool:
        return self.kind == "rf" and (self.v_c_V != 0.0 or self.dE_ref_eV != 0.0)


def _sweep(e_rf: list[float], e_dc: list[float], *, n: int, ds: float, w0_eV: float,
           mass_eV: float, charge: float, lam_m: float, rf_phase_deg: float,
           phi0_deg: float) -> dict:
    """One pass of HELIX ``advance_ref`` over pre-sampled on-axis fields.

    ``e_rf`` / ``e_dc`` are ``k·E_z(z_i)/Norm`` [V/m] at the ``n`` midpoints, split into the
    RF part (phasor ``cos φ``) and the static part (phasor 1); the probe pass of
    ``_calibrate_sync_phase`` passes everything as ``e_rf`` (HELIX's probe applies the cos
    phasor to every electric channel).  ``re``/``im`` are the complex voltage referred to
    the map entrance, so ``dE = re·cos φ_in − im·sin φ_in`` with ``φ_in = rf_phase + φ0``.
    """
    deg = math.pi / 180.0
    w = w0_eV
    phi = phi0_deg
    re = im = dE = dE_dc = 0.0
    beta = _beta_of(w, mass_eV)
    rf_on = lam_m > 0.0
    for i in range(n):
        dphi_half = 180.0 * ds / (beta * lam_m) if (rf_on and beta > 0.0) else 0.0
        phi_mid = phi + dphi_half
        internal = (phi_mid - phi0_deg) * deg
        dc = charge * e_dc[i] * ds
        dw = charge * e_rf[i] * math.cos((rf_phase_deg + phi_mid) * deg) * ds + dc
        dE += dw
        dE_dc += dc
        re += charge * e_rf[i] * math.cos(internal) * ds
        im += charge * e_rf[i] * math.sin(internal) * ds
        w += dw
        beta = _beta_of(w, mass_eV)
        if rf_on and beta > 0.0:
            phi += 360.0 * ds / (beta * lam_m)
    return {"re": re, "im": im, "dE": dE, "dE_static": dE_dc, "w": w,
            "phase_advance_deg": phi - phi0_deg}


def _calibrate_sync_offset(e_all: list[float], theta_deg: float, sweep_kw: dict,
                           tol_deg: float = 0.005, max_iter: int = 30) -> float:
    """ψ such that the deck's θ is the *synchronous* phase — HELIX ``_calibrate_sync_phase``
    (fixed point of ``ψ = arg V_c(θ − ψ)``, 0.7 under-relaxation, 0.005° tolerance, probed at
    the operating point so the β trajectory is the one the run itself produces)."""
    zeros = [0.0] * len(e_all)
    psi = 0.0
    for _ in range(max_iter):
        acc = _sweep(e_all, zeros, rf_phase_deg=theta_deg - psi, phi0_deg=0.0, **sweep_kw)
        psi_new = math.degrees(math.atan2(acc["im"], acc["re"]))
        if abs(psi_new - psi) < tol_deg:
            return psi_new
        psi = 0.3 * psi + 0.7 * psi_new
    return psi


def integrate_map(fm, ref: ReferenceParticle, data: FieldMapData, *,
                  entrance_phase_deg: float = 0.0, n_steps: int | None = None,
                  with_checksums: bool = False) -> MapSummary:
    """Integrate one :class:`~lattix.ir.elements.FieldMap` at reference state *ref*.

    *fm* supplies the card's length, ``ke``/``kb``, phase and ``p_flag``; *data* the field.
    ``entrance_phase_deg`` is the reference particle's running RF phase φ_s at the map
    entrance — it only matters for a map whose phase is *not* a synchronous phase
    (``p_flag = 0`` and no ``SET_SYNC_PHASE``), where TraceWin/HELIX add it to the deck θ.
    """
    length = float(fm.length)
    z_axis = data.z
    z_start = float(z_axis[0]) if len(z_axis) else 0.0
    if length <= _L_TOL and len(z_axis) > 1:
        length = float(z_axis[-1] - z_axis[0])
    n = int(n_steps or (fm.tracking or {}).get("n_steps") or max(len(z_axis) - 1, 50))
    n = max(n, 1)
    ds = length / n
    z_mid = z_start + (np.arange(n) + 0.5) * ds

    charge = float(ref.species.charge)
    mass = float(ref.species.mass_eV)
    sync = bool(getattr(fm.rf, "phase_is_sync", False) or int(getattr(fm, "p_flag", 0) or 0) == 1)
    theta_deg = tracewin_phase_deg(float(fm.rf.phase_rad), ref.species.charge, sync)

    # RF clock: an electric map propagates its own frequency to the reference (HELIX
    # ``_propagate_frequency_to_ref``); a static map leaves the machine clock alone.
    freq = fm.rf.frequency_Hz if (data.has_electric and fm.rf.frequency_Hz) else ref.rf_frequency_Hz
    lam = C_LIGHT / freq if freq else 0.0

    # on-axis electric field at the quadrature nodes, split by phasor
    e_rf = np.zeros(n)
    e_dc = np.zeros(n)
    n_e = 0
    for ch in data.ordered:
        if not ch.is_electric or ch.digit == 9:
            continue
        prof = ch.on_axis()
        if prof is None:
            continue
        n_e += 1
        sampled = np.interp(z_mid, prof[0], prof[1], left=0.0, right=0.0) * (float(fm.ke) / ch.norm)
        if ch.is_static:
            e_dc += sampled
        else:
            e_rf += sampled
    l_rf, l_dc = e_rf.tolist(), e_dc.tolist()

    sweep_kw = dict(n=n, ds=ds, w0_eV=float(ref.kinetic_energy_eV), mass_eV=mass,
                    charge=charge, lam_m=lam)
    psi = 0.0
    if n_e and sync:
        psi = _calibrate_sync_offset((e_rf + e_dc).tolist(), theta_deg, sweep_kw)
    rf_phase_deg = theta_deg - psi if sync else theta_deg
    phi0_deg = 0.0 if sync else float(entrance_phase_deg)
    acc = _sweep(l_rf, l_dc, rf_phase_deg=rf_phase_deg, phi0_deg=phi0_deg, **sweep_kw)

    phi_in_deg = rf_phase_deg + phi0_deg
    v_c = math.hypot(acc["re"], acc["im"])
    phase_sync = math.radians(phi_in_deg + math.degrees(math.atan2(acc["im"], acc["re"]))) if v_c else 0.0

    # fixed-β transit-time factor and PLAN's V_eff = ∫|E_z|·T at the *entrance* β
    beta_in = _beta_of(float(ref.kinetic_energy_eV), mass)
    e_tot = e_rf + e_dc
    int_ez = float(np.sum(e_tot) * ds)
    int_abs = float(np.sum(np.abs(e_tot)) * ds)
    k_wave = 2.0 * math.pi / (beta_in * lam) if (lam > 0.0 and beta_in > 0.0) else 0.0
    phase = k_wave * (z_mid - z_start)
    v_eff = float(abs(np.sum(e_tot * np.exp(1j * phase))) * ds)
    ttf = v_eff / int_abs if int_abs else 1.0

    # static magnetic channels: ∫B, ∫B², hard-edge equivalent, quadrupole ∫G.  An RF
    # magnetic channel is a cavity's own B_θ/B_z, never a solenoid, so it is left out.
    int_b = int_b2 = 0.0
    int_g = int_g2 = None
    quad = False
    for ch in data.ordered:
        if ch.is_electric or not ch.is_static:
            continue
        prof = ch.on_axis()
        if prof is None:
            continue
        zz, ff = prof
        f = ff * (float(fm.kb) / ch.norm)
        if ch.digit == 9:
            quad = True
            int_g = (int_g or 0.0) + _range_integral(zz, f, z_start, z_start + length)
            int_g2 = (int_g2 or 0.0) + _range_integral(zz, f * f, z_start, z_start + length)
            continue
        int_b += _range_integral(zz, f, z_start, z_start + length)
        int_b2 += _range_integral(zz, f * f, z_start, z_start + length)
    l_eff = (int_b * int_b) / int_b2 if int_b2 else 0.0
    b_eff = int_b2 / int_b if int_b else 0.0

    kind = "rf" if n_e else ("quad" if quad else ("solenoid" if int_b2 else "none"))
    files = tuple(data.files)
    return MapSummary(
        kind=kind, length_m=length, n_steps=n, frequency_Hz=freq,
        beta_in=beta_in, beta_out=_beta_of(acc["w"], mass),
        int_Ez_V=int_ez, int_abs_Ez_V=int_abs, ttf=ttf, v_eff_V=v_eff, v_c_V=v_c,
        phase_rf_rad=math.radians(phi_in_deg), phase_sync_rad=phase_sync,
        sync_offset_deg=psi, dE_ref_eV=acc["dE"], dE_static_eV=acc["dE_static"],
        phase_advance_deg=acc["phase_advance_deg"],
        int_Bz_Tm=int_b, int_Bz2_T2m=int_b2, L_eff_m=l_eff, B_eff_T=b_eff,
        int_Gz_Tm_per_m=int_g, int_Gz2_T2m_per_m2=int_g2, files=files,
        sha256=tuple(sha256_file(f) for f in files) if with_checksums else (),
    )


# ---------------------------------------------------------------------------
# Lattice-level annotation (the TraceWin reader's FieldMap post-pass)
#: EXACT — the map's files were read and its reference gain integrated
FM_INTEGRATED = "FM_INTEGRATED"
#: EXACT — the card converted losslessly but its files could not be decoded, so ``dE_ref``
#: stays unknown (the loss only materialises where a writer degrades the map, which reports
#: its own LOSSY code); the reason is also appended to ``Lattice.warnings``
FM_NOT_INTEGRATED = "FM_NOT_INTEGRATED"


def load_field_map(el) -> FieldMapData | None:
    """:class:`FieldMapData` for a ``FieldMap`` element the reader resolved on disk.

    ``None`` when the element carries no ``geom``/directory/file name or a component file is
    missing; the file-reading exceptions propagate so the caller can report them.
    """
    geom = getattr(el, "geom", None)
    meta = el.meta or {}
    map_dir = meta.get("field_map_dir")
    if geom is None or map_dir is None or not getattr(el, "files", None):
        return None
    if meta.get("field_files_missing"):
        return None
    return FieldMapData.load(int(geom), map_dir, el.files[0])


def _mark(report, element: str, kind: str, cls, code: str, message: str, **details) -> None:
    """Re-label the element's own ledger entry (one entry per source element, PLAN §4.4)."""
    if report is None:
        return
    from lattix.fidelity import FidelityClass

    for entry in reversed(report.entries):
        if entry.element == element and entry.kind == kind:
            entry.cls = FidelityClass(cls)
            entry.code = code
            entry.message = message
            entry.details.update(details)
            return
    report.add(cls, code, message, element=element, kind=kind, **details)


def _not_integrated(el, report, warnings, why: str) -> None:
    from lattix.fidelity import FidelityClass

    msg = f"field map {el.name!r} not integrated: {why}; dE_ref unknown"
    _mark(report, el.name, "FieldMap", FidelityClass.EXACT, FM_NOT_INTEGRATED, msg)
    if warnings is not None:
        warnings.append(msg)


def _annotate_one(el, ref: ReferenceParticle, phase_deg: float, report, checksums: bool,
                  warnings: list[str] | None = None) -> MapSummary | None:
    """Integrate one FieldMap and write the result onto the element."""
    from lattix.fidelity import FidelityClass

    try:
        data = load_field_map(el)
    except Exception as exc:                                  # noqa: BLE001 — reported, never raised
        _not_integrated(el, report, warnings, f"{type(exc).__name__}: {exc}")
        return None
    if data is None:
        return None
    if not data.channels:
        _not_integrated(el, report, warnings, f"geom {getattr(el, 'geom', None)} selects no field channel")
        return None
    if len(data.z) < 2:
        _not_integrated(el, report, warnings, "fewer than two samples along z")
        return None
    try:
        s = integrate_map(el, ref, data, entrance_phase_deg=phase_deg, with_checksums=checksums)
    except Exception as exc:                                  # noqa: BLE001
        _not_integrated(el, report, warnings, f"{type(exc).__name__}: {exc}")
        return None
    el.rf.dE_ref_eV = s.dE_ref_eV
    if s.kind == "rf":
        el.rf.voltage_V = s.v_eff_V
        el.rf.ttf = s.ttf
    el.meta["map_summary"] = s.to_dict()
    if checksums and s.sha256:
        el.meta["field_files_sha256"] = dict(zip(s.files, s.sha256, strict=True))
    _mark(report, el.name, "FieldMap", FidelityClass.EXACT, FM_INTEGRATED, _message(s),
          **_details(s))
    return s


def _message(s: MapSummary) -> str:
    if s.kind == "rf":
        return (f"field map integrated: dE_ref = {s.dE_ref_eV:.6g} eV, V_eff = {s.v_eff_V:.6g} V, "
                f"T({s.beta_in:.4f}) = {s.ttf:.6g}, φs = {math.degrees(s.phase_sync_rad):.4g}°")
    if s.kind == "solenoid":
        return (f"static solenoid map integrated: ∫B = {s.int_Bz_Tm:.6g} T·m, "
                f"∫B² = {s.int_Bz2_T2m:.6g} T²·m, L_eff = {s.L_eff_m:.6g} m, B_eff = {s.B_eff_T:.6g} T")
    if s.kind == "quad":
        return f"static quadrupole map integrated: ∫G dz = {s.int_Gz_Tm_per_m:.6g} T"
    return "field map integrated: no field channel contributes to the reference"


def _details(s: MapSummary) -> dict:
    keep = ("dE_ref_eV", "v_eff_V", "v_c_V", "ttf", "int_Ez_V", "int_Bz_Tm", "int_Bz2_T2m",
            "L_eff_m", "B_eff_T", "int_Gz_Tm_per_m", "int_Gz2_T2m_per_m2")
    return {"map_kind": s.kind} | {k: getattr(s, k) for k in keep}


def annotate_field_maps(lat, *, use: str | None = None, report=None, checksums: bool = True,
                        warnings: list[str] | None = None) -> list[tuple[str, MapSummary]]:
    """Integrate every field map in *lat* at its own reference energy and record the result.

    Walks the expanded line keeping the reference particle and its running RF phase φ_s (the
    same rule ``walk.propagate`` uses, plus HELIX's φ_s advance so a non-synchronous map sees
    the phase TraceWin would give it).  Each :class:`~lattix.ir.elements.FieldMap` gets
    ``rf.dE_ref_eV``, ``rf.voltage_V`` (= ``V_eff``), ``rf.ttf`` and ``meta['map_summary']``;
    a :class:`~lattix.ir.elements.Superposition` cluster gets the sum of its children's gains,
    each child integrated in turn at the energy the previous ones left behind.
    """
    from lattix.fidelity import FidelityClass
    from lattix.ir.elements import FieldMap, Freq, Superposition
    from lattix.ir.walk import energy_gain_eV

    ref = lat.reference
    phase_deg = 0.0
    if warnings is None:
        warnings = lat.warnings
    out: list[tuple[str, MapSummary]] = []
    for p in lat.flatten(use):
        el = p.element
        if isinstance(el, Freq):
            ref = ref.advanced(rf_frequency_Hz=el.frequency_Hz)
            continue
        advance = None
        if isinstance(el, FieldMap):
            s = _annotate_one(el, ref, phase_deg, report, checksums, warnings)
            if s is not None:
                out.append((el.name, s))
                advance = s.phase_advance_deg
        elif isinstance(el, Superposition):
            total = 0.0
            inner = ref
            kids: list[tuple[str, MapSummary]] = []
            for _offset, child_name in el.children:
                child = lat.elements.get(child_name)
                if not isinstance(child, FieldMap):
                    continue
                s = _annotate_one(child, inner, phase_deg, report, checksums, warnings)
                if s is None:
                    continue
                kids.append((child_name, s))
                total += s.dE_ref_eV
                inner = inner.advanced(dE_eV=s.dE_ref_eV)
            if kids:
                el.rf.dE_ref_eV = total
                el.meta["map_summary"] = {"kind": "cluster", "dE_ref_eV": total,
                                          "children": [n for n, _ in kids]}
                _mark(report, el.name, "Superposition", FidelityClass.EXACT, FM_INTEGRATED,
                      f"superposed field maps integrated in order: dE_ref = {total:.6g} eV",
                      dE_ref_eV=total, children=len(kids))
                out.extend(kids)
        dE = energy_gain_eV(el, ref)
        if advance is None:
            lam = ref.wavelength_m
            advance = (360.0 * el.length / (ref.beta * lam)) if (lam and el.length and ref.beta) else 0.0
        phase_deg += advance
        rf = getattr(el, "rf", None)
        f_new = rf.frequency_Hz if (rf is not None and rf.frequency_Hz) else None
        ref = ref.advanced(dE_eV=dE, ds_m=el.length, rf_frequency_Hz=f_new)
    return out


# ---------------------------------------------------------------------------
# The degradation ladder for targets that cannot carry a field map (PLAN §4.3)
#: EQUIVALENT (LOSSY without a known ``dE_ref``) — RF map → equivalent accelerating cavity
FM_TO_CAVITY = "FM_TO_CAVITY"
#: EQUIVALENT — static solenoid map → hard-edge solenoid preserving ∫B and ∫B²
FM_SOL_HARDEDGE = "FM_SOL_HARDEDGE"
#: EQUIVALENT — static quadrupole map (geom digit 9) → hard-edge quadrupole preserving ∫G, ∫G²
FM_QUAD_HARDEDGE = "FM_QUAD_HARDEDGE"
#: LOSSY — nothing is known about the map, so only its length survives
FM_TO_DRIFT = "FM_TO_DRIFT"

_COS_TOL = 1e-9


@dataclass(frozen=True)
class Replacement:
    """What a :class:`~lattix.ir.elements.FieldMap` becomes in a format without field maps.

    ``parts`` are consecutive elements whose lengths sum to the map's own length, so the
    survey never moves: a cavity is one full-length element (a MAD-X ``rfcavity``, an
    elegant ``RFCA`` and a MAD8 ``RFCAVITY`` are all drift → thin kick at the centre →
    drift), while a hard-edge magnet is padded with drifts to stay centred in the map.
    """

    code: str
    cls: str                       # EQUIVALENT | LOSSY
    message: str
    parts: tuple = ()
    details: dict = field(default_factory=dict)
    extra: tuple[tuple[str, str, str], ...] = ()     # further (cls, code, message) entries

    @property
    def main(self):
        """The part that carries the physics (the padding drifts are the others)."""
        return max(self.parts, key=lambda e: (e.kind != "Drift", e.length))

    @property
    def padded(self) -> bool:
        return len(self.parts) > 1


def replacement_for(el, *, thin_cavity: bool = False) -> Replacement:
    """Degrade one field map for a target that has no field-map element.

    ``thin_cavity=True`` puts a zero-length cavity at the map centre between two ``L/2``
    drifts (Bmad, whose ``lcavity`` must be ``l = 0, cavity_type = traveling_wave`` to be
    thin at all — docs/oracles.md); the default keeps one full-length cavity element.
    """
    from lattix.ir.elements import RFP, Drift, MagneticMultipoleP, Quadrupole, RFCavity, Solenoid, SolenoidP

    s = (el.meta or {}).get("map_summary") or {}
    kind = s.get("kind")
    length = float(el.length)
    rf = el.rf
    extra: list[tuple[str, str, str]] = []

    if kind in ("solenoid", "quad"):
        if kind == "solenoid":
            l_eff, strength = float(s.get("L_eff_m") or 0.0), float(s.get("B_eff_T") or 0.0)
            int_1, int_2 = s.get("int_Bz_Tm"), s.get("int_Bz2_T2m")
            code = FM_SOL_HARDEDGE
            msg = (f"static solenoid map → hard-edge solenoid L_eff = {l_eff:.6g} m, "
                   f"B_eff = {strength:.6g} T (∫B = {int_1:.6g} T·m and ∫B² = {int_2:.6g} T²·m preserved), "
                   "centred in the map with drift padding")
            body = Solenoid(name=el.name, length=min(l_eff, length),
                            solenoid=SolenoidP(Bsol_T=strength))
        else:
            int_1 = s.get("int_Gz_Tm_per_m") or 0.0
            int_2 = s.get("int_Gz2_T2m_per_m2") or 0.0
            l_eff = (int_1 * int_1 / int_2) if int_2 else length
            strength = (int_2 / int_1) if int_1 else 0.0
            code = FM_QUAD_HARDEDGE
            msg = (f"static quadrupole map → hard-edge quadrupole L_eff = {l_eff:.6g} m, "
                   f"G = {strength:.6g} T/m (∫G = {int_1:.6g} T and ∫G² = {int_2:.6g} T²/m preserved), "
                   "centred in the map with drift padding")
            body = Quadrupole(name=el.name, length=min(l_eff, length),
                              multipole=MagneticMultipoleP(Bn={1: strength}))
        body.aperture = el.aperture
        body.provenance = el.provenance
        return Replacement(code=code, cls="EQUIVALENT", message=msg,
                           parts=_centre(el, body, length),
                           details={"L_eff_m": l_eff, "strength": strength,
                                    "integral": int_1, "integral_sq": int_2})

    volt, phase, dE = _cavity_numbers(s, rf)
    if volt is not None:
        known = dE is not None
        if kind == "rf" and float(s.get("int_Bz2_T2m") or 0.0):
            extra.append(("LOSSY", "FM_STATIC_B_DROPPED",
                          "the map's static magnetic channel has no place on an RF cavity; "
                          f"∫B = {s.get('int_Bz_Tm'):.6g} T·m dropped"))
        cav = RFCavity(
            name=el.name, length=0.0 if thin_cavity else length,
            aperture=el.aperture, provenance=el.provenance,
            rf=RFP(frequency_Hz=rf.frequency_Hz or s.get("frequency_Hz"),
                   voltage_V=volt, phase_rad=phase, dE_ref_eV=dE, L_active_m=length,
                   phase_is_sync=True, cavity_type="TRAVELING_WAVE" if thin_cavity else "STANDING_WAVE"),
        )
        msg = (f"field map → {'thin' if thin_cavity else 'thick'} cavity, "
               f"V = {volt:.6g} V at φs = {math.degrees(phase):.4g}°"
               + (f", dE_ref = {dE:.6g} eV" if known else "; its reference gain is unknown"))
        parts = _centre(el, cav, length) if thin_cavity else (cav,)
        return Replacement(code=FM_TO_CAVITY, cls="EQUIVALENT" if known else "LOSSY", message=msg,
                           parts=parts, extra=tuple(extra),
                           details={"voltage_V": volt, "phase_rad": phase, "dE_ref_eV": dE,
                                    "ttf": s.get("ttf"), "v_eff_V": s.get("v_eff_V"),
                                    "int_Ez_V": s.get("int_Ez_V")})

    return Replacement(code=FM_TO_DRIFT, cls="LOSSY",
                       message="field map replaced by a drift of the same length "
                               "(its reference gain is unknown)",
                       parts=(Drift(name=el.name, length=length, aperture=el.aperture,
                                    provenance=el.provenance),),
                       details={"files": list(getattr(el, "files", []) or [])})


def _cavity_numbers(s: dict, rf) -> tuple[float | None, float, float | None]:
    """``(voltage, synchronous phase, dE_ref)`` for the equivalent cavity, or ``(None, …)``.

    With a map summary the pair (``v_c_V``, ``phase_sync_rad``) reproduces the integrated
    gain *identically* — ``dE_ref = v_c·cos φs`` by construction — at any phase, including
    the ±90° of a buncher where ``dE_ref/cos φ`` is meaningless.
    """
    if s.get("kind") == "rf":
        v = float(s.get("v_c_V") or 0.0) or float(s.get("v_eff_V") or 0.0)
        return v, float(s.get("phase_sync_rad") or 0.0), float(s.get("dE_ref_eV") or 0.0)
    dE = rf.dE_ref_eV
    cos = math.cos(rf.phase_rad)
    if dE is not None and abs(cos) > _COS_TOL:
        return dE / cos, rf.phase_rad, dE
    if rf.voltage_V:
        return rf.voltage_V, rf.phase_rad, dE
    return None, rf.phase_rad, dE


def _centre(el, body, length: float) -> tuple:
    """*body* centred inside a map of *length*, padded with drifts (dropped when zero)."""
    from lattix.ir.elements import Drift

    pad = max(0.0, (length - float(body.length)) / 2.0)
    if pad <= 0.0:
        return (body,)
    return (Drift(name=f"{el.name}_D1", length=pad), body, Drift(name=f"{el.name}_D2", length=pad))
