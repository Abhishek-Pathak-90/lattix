"""IMPACT-Z oracle: per-element 6×6 maps and reference energies from ``ImpactZexe``.

How the numbers are obtained (everything below MEASURED on this machine 2026-09-03 with
conda-forge ``impact-z`` 2.7.7, osx-arm64):

* IMPACT-Z insists on reading a file called ``ImpactZ.in`` from its working directory,
  so the adapter copies the deck (and any ``rfdata*.in`` next to it) into a temp dir.
* The deck is re-emitted with ``flagdist = 19``, which makes ``readin_Dist``
  (``Distribution.f90:1666-1713``) read ``particle.in``: one line with the particle
  count, then nine columns per particle in IMPACT-Z's **internal** units, taken
  verbatim.  The adapter writes a 13-particle probe — the reference plus ``±h`` in each
  of the six coordinates — so every element's Jacobian comes out of one run.
* A zero-length ``-2`` card is inserted before the first element and after every element.
  ``phase_Output`` (``Output.f90:1776-1855``) then writes the same nine columns to
  ``fort.<mapstp>``; with ``bnseg = 0`` these cards cost no integration step and add no
  ``fort.18`` row.  ``R_elem[i] = J_i · J_{i-1}⁻¹`` with ``J_k`` the central-difference
  Jacobian at dump *k*.
* ``fort.18`` (``Output.f90:341``) is written once before tracking and once per
  integration step, so its rows are ``1 + Σ bnseg``; its columns are
  ``z [m], reference phase [rad], γ, kinetic energy [MeV], β, max radius [m]``.
  The adapter turns column 4 into ``ref_kinetic_eV_in/out`` per element.

Native basis (``BeamBunch.f90:276-296, 483-502``; ``PhysConst.f90:30``)
----------------------------------------------------------------------
``(x/λ̄, γβ_x, y/λ̄, γβ_y, ω·(t − t_ref) [rad], γ_ref − γ)`` with ``λ̄ = Scxl = c/(2πf)``.
The 5th coordinate is **late-positive** and the 6th is the **negative** energy deviation
(``pz = sqrt((γ0 − q6)² − 1 − …)``).  The map onto the common basis
``(x [m], px/p0, y [m], py/p0, z [m] ahead-positive, δ)`` is therefore

``T = diag(λ̄, 1/βγ, λ̄, 1/βγ, −βλ̄, −1/β²γ)``

which is what :func:`transform_to_common` returns.  **``lattix.oracles.basis`` does not
agree**: its ``Basis.IMPACTZ`` branch leaves ``d[0] = d[2] = 1`` (so ``x`` stays in units
of ``λ̄``: a 1 m drift comes out with ``R12 = 50.88`` instead of 1) and uses
``d[5] = +1/β²γ`` (so ``R56`` and the whole dispersion column come out with the wrong
sign).  Until that is fixed the adapter converts the matrices itself and returns
``Basis.COMMON`` by default; ``basis="native"`` returns the raw internal maps.

Verification of the transform (MEASURED, ``examples`` in the module tests):

* 1 m drift at 2.1 MeV → ``R56 = +0.99553867`` in the common basis, the value
  ``docs/oracles.md`` pins for MAD-X, xtrack, Bmad, TraceWin and HELIX (0.99553867
  against 0.99553867, 2e-9 — the floor set by IMPACT-Z's low-β arithmetic, see
  :data:`DEFAULT_STEPS`);
* 0.3 m quadrupole, G = 5 T/m → the analytic thick-quad map to **4e-14** (``map_steps``
  ≥ 20; with ``map_steps = 1`` the drift-kick splitting costs 1.2e-4);
* 1 m sector bend, θ = 0.1 rad → MAD-X's analytic map to **1.1e-12**;
* 0.4 m solenoid, B = 0.5 T → the analytic solenoid map to **4.6e-15** (the
  "The linear map does NOT work" comment at ``Sol.f90:9`` is stale for ``flagmap = 1``);
* thin cavity, 1 MV at φ_s = −30° through IMPACT-Z's ideal-cavity model → ΔE
  = 866 025.403 784 337 eV (1.2e-16 relative to ``V·cos φ_s``), ``R65 = −4.30458``
  (Bmad's ``lcavity`` and HELIX give −4.305) and ``det(2×2) = p_in/p_out`` to 5e-12.

What this oracle cannot give
----------------------------
IMPACT-Z computes no Twiss functions, no dispersion and no floor coordinates, so
``twiss``, ``disp`` and ``survey`` are empty/``None``.  ``fort.24/25/26`` hold rms
envelopes only (``meta["envelopes"]`` when ``envelopes=True``).  Nothing here is
available for the ``flagmap = 2`` Lorentz integrator's multipoles, whose linear-map
counterpart is broken (see :mod:`lattix.formats.impactz.writer`).

Binary lookup: ``$IMPACTZ_EXE``, then ``$LATTIX_ENV_BIN/ImpactZexe``, then
``~/anaconda3/envs/lattix/bin/ImpactZexe``, then ``ImpactZexe``/``ImpactZexeMac`` on PATH.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from lattix.oracles.base import C_LIGHT, Basis, BeamSpec, OracleResult, Probe, register

_DEFAULT_ENV_BIN = Path("~/anaconda3/envs/lattix/bin").expanduser()

#: Fortran unit numbers IMPACT-Z uses itself — a ``-2`` dump must not land on one
#: (``Output.f90`` writes 18, 24-30, 32; ``Data.f90`` opens 13/14/15; ``-6`` writes 8).
_RESERVED_UNITS = frozenset({6, 8, 12, 13, 14, 15, 18, *range(24, 33)})

#: first ``fort.N`` used for the probe dumps (well clear of ``_RESERVED_UNITS``).
FIRST_DUMP_UNIT = 1001

#: Default central-difference steps, in the **native** coordinates
#: ``(x/Scxl, γβ_x, y/Scxl, γβ_y, φ [rad], γ)``.  Chosen by a convergence scan
#: (MEASURED 2026-09-03, proton at 2.1 MeV): the transverse block reaches the analytic
#: thick-quadrupole map to 2.8e-14 over four decades of step size, while the
#: longitudinal ``R56`` bottoms out at ~2e-9 around ``h6 = 1e-7`` — below that,
#: IMPACT-Z's own ``sqrt((γ0 − q6)² − 1)`` loses the digits (``γ² − 1 = 4.5e-3`` at
#: β = 0.067), above it the truncation error grows as h².  Raise ``h6`` at high energy.
DEFAULT_STEPS = (1e-6, 1e-7, 1e-6, 1e-7, 1e-6, 1e-7)


def find_impactz() -> Path | None:
    """``$IMPACTZ_EXE`` → ``$LATTIX_ENV_BIN`` → ``~/anaconda3/envs/lattix/bin`` → PATH."""
    v = os.environ.get("IMPACTZ_EXE")
    if v:
        p = Path(v).expanduser()
        return p if p.is_file() else None
    cands = []
    v = os.environ.get("LATTIX_ENV_BIN")
    if v:
        cands.append(Path(v).expanduser() / "ImpactZexe")
    cands.append(_DEFAULT_ENV_BIN / "ImpactZexe")
    for p in cands:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    for name in ("ImpactZexe", "ImpactZexeMac"):
        w = shutil.which(name)
        if w:
            return Path(w)
    return None


@functools.lru_cache(maxsize=8)
def impactz_version(exe: str) -> str:
    """Version string from IMPACT-Z's banner (it only prints it once it starts reading
    a deck, so a throw-away one-drift deck is fed to it in a temp dir)."""
    from lattix.formats.impactz.reader import Header

    with tempfile.TemporaryDirectory(prefix="lattix_impactz_ver_") as td:
        h = Header(np=1, kinetic_energy_eV=1e6, mass_eV=938.272088e6, charge=1.0,
                   frequency_Hz=1e9)
        (Path(td) / "ImpactZ.in").write_text(
            _header_text(h, flagdist=2, np_=1, current=0.0) + "0 0 0 -99 /\n")
        r = subprocess.run([exe], cwd=td, capture_output=True, text=True, timeout=120,
                           stdin=subprocess.DEVNULL)
    for ln in (r.stdout + r.stderr).splitlines():
        if "IMPACT-Z" in ln and "Version" in ln:
            return ln.strip(" !").strip()
    return "unknown"


def transform_to_common(kinetic_eV: float, mass_eV: float, frequency_Hz: float) -> np.ndarray:
    """MEASURED 6×6 ``T`` with ``x_common = T @ x_native`` (see the module docstring)."""
    if not frequency_Hz:
        raise ValueError("the IMPACT-Z basis needs the deck's reference frequency")
    g = 1.0 + kinetic_eV / mass_eV
    b = float(np.sqrt(1.0 - 1.0 / (g * g)))
    xl = C_LIGHT / (2.0 * np.pi * frequency_Hz)          # Scxl (PhysConst.f90:30)
    return np.diag([xl, 1.0 / (b * g), xl, 1.0 / (b * g), -b * xl, -1.0 / (b * b * g)])


# ---------------------------------------------------------------------------
# deck generation
# ---------------------------------------------------------------------------
def _g17(x: float) -> str:
    return f"{float(x):.17g}"


def _header_text(h, *, flagdist: int, np_: int, current: float) -> str:
    """Re-emit the eleven header records with the probe's distribution settings.

    Lines 8-10 must carry finite, positive Twiss triplets even for ``flagdist = 19``:
    ``sample_Dist`` converts them before it dispatches (``Distribution.f90:62-86``) and
    ``construct_CompDom`` uses them for the initial mesh.
    """
    dist = "0 1 1e-06 1 1 0 0"
    return "\n".join([
        f"{h.npcol} {h.nprow}",
        f"{h.dim} {np_} {h.flagmap} {h.flagerr} 1",
        f"{h.nx} {h.ny} {h.nz} {h.flagbc} {_g17(h.xrad)} {_g17(h.yrad)} {_g17(h.perdlen)}",
        f"{flagdist} 0 0 1",
        f"{np_}",
        _g17(current),
        _g17(h.charge / h.mass_eV if h.mass_eV else 0.0),
        dist, dist, dist,
        f"{_g17(current)} {_g17(h.kinetic_energy_eV)} {_g17(h.mass_eV)} "
        f"{_g17(h.charge)} {_g17(h.frequency_Hz)} {_g17(h.phase_ini_rad)}",
    ]) + "\n"


def _card_text(c) -> str:
    vals = " ".join(_g17(v) for v in c.values)
    head = f"{_g17(c.length)} {c.nseg} {c.mapstp} {c.itype}"
    return f"{head} {vals} /" if vals else f"{head} /"


def _probe_particles(steps, qmcc: float, extra: np.ndarray | None) -> tuple[str, int]:
    """13 finite-difference particles (+ an optional user probe) in internal units."""
    rows = [np.zeros(6)]
    for k in range(6):
        for sgn in (+1.0, -1.0):
            v = np.zeros(6)
            v[k] = sgn * steps[k]
            rows.append(v)
    if extra is not None:
        rows.extend(np.asarray(extra, dtype=float))
    out = [str(len(rows))]
    for i, r in enumerate(rows):
        out.append(" ".join(_g17(v) for v in r) + f" {_g17(qmcc)} 0 {i + 1}")
    return "\n".join(out) + "\n", len(rows)


# ---------------------------------------------------------------------------
@register
class ImpactzOracle:
    """IMPACT-Z adapter (``ImpactZexe``)."""

    name = "impactz"
    formats = ("impactz",)
    basis = Basis.IMPACTZ
    timeout_s = 900

    def __init__(self, exe: Path | str | None = None):
        self._exe_override = Path(exe).expanduser() if exe else None

    # ------------------------------------------------------------------
    def exe(self) -> Path:
        p = self._exe_override or find_impactz()
        if p is None:
            raise FileNotFoundError(
                "ImpactZexe not found: set IMPACTZ_EXE or LATTIX_ENV_BIN, or install it "
                f"(conda install -c conda-forge impact-z; {_DEFAULT_ENV_BIN}/ImpactZexe, PATH)")
        return p

    def available(self) -> tuple[bool, str]:
        try:
            exe = self.exe()
        except FileNotFoundError as e:
            return False, str(e)
        try:
            ver = impactz_version(str(exe))
        except Exception as e:                              # noqa: BLE001
            return False, f"{exe} did not run: {e}"
        return True, f"{exe} ({ver})"

    # ------------------------------------------------------------------
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            maps: bool = True, basis: str = "common", steps=DEFAULT_STEPS,
            envelopes: bool = False, keep_workdir: bool = True) -> OracleResult:
        """Run IMPACT-Z on *deck* (an ``ImpactZ.in``).

        ``maps=False`` skips the probe entirely and returns identity ``R_elem`` with
        ``meta["maps"] = "not requested"`` — use it when only the reference energy
        profile is wanted.  ``basis`` is ``"common"`` (default, using
        :func:`transform_to_common`) or ``"native"`` (IMPACT-Z's internal coordinates,
        ``Basis.IMPACTZ``).  ``steps`` are the central-difference steps in the native
        coordinates.
        """
        from lattix.formats.impactz.reader import parse_deck

        if fmt not in (None, "impactz"):
            raise ValueError(f"IMPACT-Z reads ImpactZ.in decks only, not {fmt!r}")
        if basis not in ("common", "native"):
            raise ValueError(f"basis must be 'common' or 'native', got {basis!r}")
        exe = self.exe()
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        wd = Path(workdir).resolve() if workdir else Path(tempfile.mkdtemp(prefix="lattix_impactz_"))
        wd.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        header, cards, _ = parse_deck(deck.read_text(errors="replace"))
        cards = [c for c in cards if c.itype != -99]
        if not cards:
            raise RuntimeError(f"{deck} has no beam-line elements")
        for f in sorted(deck.parent.glob("rfdata*.in")):
            shutil.copy2(f, wd / f.name)

        if beam is not None:
            header.kinetic_energy_eV = beam.kinetic_energy_eV
            header.mass_eV = beam.mass_eV
            header.charge = float(beam.charge)
            if beam.frequency_Hz:
                header.frequency_Hz = beam.frequency_Hz
            warnings.append("the BeamSpec overrode the deck's own reference particle "
                            f"({beam.species}, {beam.kinetic_energy_eV:.6g} eV)")
        if not header.frequency_Hz:
            raise RuntimeError(f"{deck}: the header has no RF frequency, so IMPACT-Z's "
                               "internal length scale c/2πf is undefined")
        if header.current_A:
            warnings.append(f"the deck's {header.current_A:g} A beam current was set to 0: "
                            "space charge makes the transfer map amplitude-dependent")
        if header.flagmap != 1:
            warnings.append("the deck asks for the nonlinear Lorentz integrator "
                            "(flagmap = 2); its maps are not the linear ones")

        native_probe = None
        if probe is not None:
            T = transform_to_common(header.kinetic_energy_eV, header.mass_eV,
                                    header.frequency_Hz)
            native_probe = np.asarray(probe.coords, dtype=float) @ np.linalg.inv(T).T
        n_probe = 0 if native_probe is None else len(native_probe)

        units = self._dump_units(len(cards)) if maps else []
        text = self._instrumented_deck(header, cards, units, maps=maps,
                                       n_particles=13 + n_probe if maps else header.np)
        (wd / "ImpactZ.in").write_text(text)
        if maps:
            pin, npt = _probe_particles(steps, header.charge / header.mass_eV, native_probe)
            (wd / "particle.in").write_text(pin)
        elif probe is not None:
            raise ValueError("maps=False cannot track a probe (it needs particle.in)")

        log = self._execute(exe, wd)
        warnings += _log_warnings(log)

        names = [self._name(c, i) for i, c in enumerate(cards)]
        lengths = np.array([c.length for c in cards], dtype=float)
        s_out = np.cumsum(lengths)
        ke_in, ke_out, phase = self._reference(wd / "fort.18", cards, header, warnings)

        R = np.tile(np.eye(6), (len(cards), 1, 1))
        probe_out = None
        meta: dict = {
            "workdir": str(wd), "deck": str(wd / "ImpactZ.in"), "impactz": str(exe),
            "impactz_version": impactz_version(str(exe)),
            "flagmap": header.flagmap, "frequency_Hz": header.frequency_Hz,
            "scxl_m": C_LIGHT / (2.0 * np.pi * header.frequency_Hz),
            "reference_phase_rad": phase,
            "native_basis": "(x/Scxl, gamma*beta_x, y/Scxl, gamma*beta_y, "
                            "omega*(t-t_ref) [rad, late-positive], gamma_ref - gamma)",
            "twiss": "not available: IMPACT-Z computes no Twiss/dispersion/floor output",
        }
        if maps:
            J, lost = self._jacobians(wd, units, steps, len(cards), warnings)
            R = _elementwise(J)
            meta["dump_units"] = units
            meta["fd_steps"] = list(steps)
            if lost:
                warnings.append(f"{lost} probe particle(s) were lost; the affected maps "
                                "are NaN")
            if native_probe is not None:
                last = _read_dump(wd / f"fort.{units[-1]}", 13 + n_probe)
                probe_out = last[13:, :6]
        else:
            meta["maps"] = "not requested (maps=False): R_elem is the identity"
            warnings.append("maps=False: R_elem is the identity, only the reference "
                            "energy profile is meaningful")
        if envelopes:
            meta["envelopes"] = self._envelopes(wd)

        out_basis = Basis.IMPACTZ
        freq = np.full(len(cards), float(header.frequency_Hz))
        if basis == "common":
            if maps:
                R = _to_common(R, ke_in, ke_out, header.mass_eV, header.frequency_Hz)
            if probe_out is not None:
                T_end = transform_to_common(float(ke_out[-1]), header.mass_eV,
                                            header.frequency_Hz)
                probe_out = probe_out @ T_end.T
            out_basis = Basis.COMMON
            # lattix.oracles.basis.Basis.IMPACTZ was corrected from this adapter's measurement
            # (2026-09-03); warn only if the two transforms ever drift apart again
            from lattix.oracles.basis import transform_matrix as _shared_transform

            _shared = _shared_transform(Basis.IMPACTZ, float(ke_in[0]), header.mass_eV, header.frequency_Hz)
            _own = transform_to_common(float(ke_in[0]), header.mass_eV, header.frequency_Hz)
            if not np.allclose(np.diag(_shared), np.diag(_own), rtol=1e-12, atol=0):
                warnings.append(
                    "matrices converted with lattix.oracles.impactz.transform_to_common: "
                    "lattix.oracles.basis's Basis.IMPACTZ branch disagrees with the measured "
                    "IMPACT-Z convention — pass basis='native' for the raw internal maps")
        if not keep_workdir and workdir is None:
            meta["workdir_removed"] = True

        return OracleResult(
            engine=self.name, basis=out_basis, names=names, length=lengths, s_out=s_out,
            R_elem=R, ref_kinetic_eV_in=ke_in, ref_kinetic_eV_out=ke_out,
            mass_eV=header.mass_eV, charge=int(round(header.charge)),
            rf_frequency_Hz=freq, probe_out=probe_out, warnings=warnings, meta=meta,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _dump_units(n_elements: int) -> list[int]:
        """One free Fortran unit per boundary (entrance + one per element)."""
        units, u = [], FIRST_DUMP_UNIT
        while len(units) < n_elements + 1:
            if u not in _RESERVED_UNITS:
                units.append(u)
            u += 1
        return units

    @staticmethod
    def _name(card, i: int) -> str:
        from urllib.parse import unquote

        from lattix.formats.impactz.reader import TYPE_NAMES

        if "name" in card.tag:
            return unquote(card.tag["name"])
        return f"{TYPE_NAMES.get(card.itype, f'type{card.itype}')}_{i + 1}"

    def _instrumented_deck(self, header, cards, units: list[int], *, maps: bool,
                           n_particles: int) -> str:
        lines = ["! instrumented by lattix.oracles.impactz — do not edit",
                 "! flagdist 19 reads particle.in; a -2 card dumps the phase space at "
                 "every element boundary"]
        lines.append(_header_text(header, flagdist=19 if maps else header.flagdist,
                                  np_=n_particles, current=0.0).rstrip("\n"))
        if maps:
            lines.append(f"0 0 {units[0]} -2 1 /")
        for i, c in enumerate(cards):
            lines.append(_card_text(c))
            if maps:
                lines.append(f"0 0 {units[i + 1]} -2 1 /")
        lines.append("0 0 0 -99 /")
        return "\n".join(lines) + "\n"

    def _execute(self, exe: Path, wd: Path) -> str:
        try:
            r = subprocess.run([str(exe)], cwd=wd, capture_output=True, text=True,
                               timeout=self.timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ImpactZexe timed out after {self.timeout_s} s in {wd}") from e
        log = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else "")
        (wd / "run.log").write_text(log)
        if r.returncode != 0 or not (wd / "fort.18").is_file():
            tail = "\n".join(log.strip().splitlines()[-30:])
            what = f"exit status {r.returncode}" if r.returncode else "no fort.18"
            raise RuntimeError(f"ImpactZexe failed ({what}) in {wd}:\n{tail}")
        return log

    # ------------------------------------------------------------------
    @staticmethod
    def _reference(fort18: Path, cards, header, warnings: list[str]):
        """``fort.18`` → per-element entry/exit kinetic energy [eV] and exit phase [rad].

        Rows are ``1 + Σ bnseg`` (one before tracking at ``AccSimulator.f90:676``, one
        per integration step at ``:1464``); columns are
        ``z, phase [rad], γ, W [MeV], β, r_max [m]``.
        """
        rows = np.atleast_2d(np.loadtxt(fort18))
        if rows.shape[1] < 5:
            raise RuntimeError(f"{fort18}: expected at least 5 columns, got {rows.shape[1]}")
        ke = rows[:, 3] * 1e6
        ph = rows[:, 1]
        want = 1 + sum(max(c.nseg, 0) for c in cards)
        if len(rows) != want:
            warnings.append(f"fort.18 has {len(rows)} rows, expected 1 + Σ bnseg = {want}; "
                            "the reference energies were aligned on the available rows")
        idx = 0
        kin, kout, pout = [], [], []
        for c in cards:
            kin.append(ke[min(idx, len(ke) - 1)])
            idx += max(c.nseg, 0)
            j = min(idx, len(ke) - 1)
            kout.append(ke[j])
            pout.append(ph[j])
        return np.array(kin), np.array(kout), np.array(pout)

    @staticmethod
    def _jacobians(wd: Path, units: list[int], steps, n_elements: int,
                   warnings: list[str]) -> tuple[np.ndarray, int]:
        J = np.empty((len(units), 6, 6))
        lost = 0
        for k, u in enumerate(units):
            pts = _read_dump(wd / f"fort.{u}", 13)
            lost += int(np.isnan(pts[:13]).any(axis=1).sum())
            for c in range(6):
                J[k, :, c] = (pts[1 + 2 * c, :6] - pts[2 + 2 * c, :6]) / (2.0 * steps[c])
        return J, lost

    @staticmethod
    def _envelopes(wd: Path) -> dict:
        """``fort.24/25/26`` rms envelopes.

        Columns (``Output.f90:342-344``): ``z [m]``, centroid, rms, momentum centroid,
        momentum rms, ``−⟨qp⟩/ε``, emittance.  x/y are in metres with ``px/(βγ)`` (i.e.
        angles) and a *normalised* emittance in m·rad; the z file is in **degrees** with
        the energy columns in **MeV** and the emittance in deg·MeV — and, because the
        6th internal coordinate is ``γ_ref − γ``, its centroid column is the *negative*
        mean energy deviation.
        """
        out = {}
        for unit, key in ((24, "x"), (25, "y"), (26, "z")):
            f = wd / f"fort.{unit}"
            if f.is_file():
                out[key] = np.atleast_2d(np.loadtxt(f))
        out["columns"] = ["z", "centroid", "rms", "p_centroid", "p_rms", "-<qp>/eps", "emit"]
        out["units"] = {"x": "m, rad, m·rad (normalised)", "y": "m, rad, m·rad (normalised)",
                        "z": "deg, MeV (centroid = -(E - E_ref)), deg·MeV"}
        return out


# ---------------------------------------------------------------------------
def _read_dump(path: Path, n_expected: int) -> np.ndarray:
    """A ``-2`` phase-space dump: nine columns, one row per surviving particle,
    ordered by MPI rank rather than by id, so rows are sorted on column 9."""
    if not path.is_file():
        raise RuntimeError(f"IMPACT-Z wrote no phase-space dump {path}; the run probably "
                           "stopped early (see run.log)")
    a = np.atleast_2d(np.loadtxt(path))
    if a.size == 0 or a.shape[1] < 9:
        raise RuntimeError(f"{path}: expected 9 columns per particle, got {a.shape}")
    out = np.full((max(n_expected, int(a[:, 8].max())), 9), np.nan)
    for row in a:
        i = int(round(row[8])) - 1
        if 0 <= i < len(out):
            out[i] = row
    return out


def _elementwise(J: np.ndarray) -> np.ndarray:
    """``R_elem[i] = J[i+1] · J[i]⁻¹`` (start→boundary Jacobians to per-element maps)."""
    n = len(J) - 1
    R = np.empty((n, 6, 6))
    for i in range(n):
        try:
            R[i] = J[i + 1] @ np.linalg.inv(J[i])
        except np.linalg.LinAlgError:                       # pragma: no cover
            R[i] = np.full((6, 6), np.nan)
    return R


def _to_common(R: np.ndarray, ke_in: np.ndarray, ke_out: np.ndarray, mass_eV: float,
               frequency_Hz: float) -> np.ndarray:
    out = np.empty_like(R)
    for i in range(len(R)):
        t_in = transform_to_common(float(ke_in[i]), mass_eV, frequency_Hz)
        t_out = transform_to_common(float(ke_out[i]), mass_eV, frequency_Hz)
        out[i] = t_out @ R[i] @ np.linalg.inv(t_in)
    return out


def _log_warnings(log: str) -> list[str]:
    out: list[str] = []
    for ln in log.splitlines():
        s = ln.strip()
        if ("out of the range" in s or "not implemented" in s or "over max" in s
                or "Not available" in s) and s not in out:
            out.append(s)
    return out
