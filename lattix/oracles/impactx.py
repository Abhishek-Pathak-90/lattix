"""ImpactX oracle: per-element 6×6 maps and reference energies by single-particle tracking.

How it runs
-----------
In-process when ``import impactx`` works; otherwise this same file is executed as a
standalone worker script (``<python> -I <this file> spec.json out.json``) under an
interpreter that has ImpactX — resolved from ``LATTIX_IMPACTX_PYTHON``, then
``LATTIX_IMPACTX_ENV`` (default ``lattix``) as a conda env, then ``conda run``, exactly
as :mod:`lattix.oracles.pytao` does for Bmad.  ``-I`` implies ``-P`` on Python ≥ 3.11, so
neither the caller's ``PYTHONPATH`` nor this file's own directory can shadow the real
``impactx`` package; the worker half below therefore imports **only** ``impactx``,
``numpy`` and the standard library.

How the maps are obtained
-------------------------
ImpactX exposes no per-element transfer map in 26.01 (``KnownElementsList.transfer_map``
is end-to-end only; ``map_trace`` appeared later), so every element is measured the way
PLAN §5.1 asks: 12 probe particles at ±h along each canonical coordinate are **re-seeded
at each element's entrance**, the element is pushed (``element.push(pc)`` advances the
reference particle first and applies **one slice**, so it is called ``nslice`` times), and
the 6×6 is the central difference.  Because ImpactX's Drift/Quad/Sbend/Sol/DipEdge pushes
*are* their analytic linear maps, this is exact to round-off (measured: ≤ 1e-15 against
MAD-X on ``fodo.madx``).  ``RefPart.kin_energy_MeV`` before and after each element gives
``ref_kinetic_eV_in/out`` — ImpactX follows the reference energy through ``ShortRF``.

Basis (MEASURED 2026-09-03, impactx 26.01, proton 2.1 MeV)
---------------------------------------------------------
ImpactX tracks ``(x, px, y, py, t, pt)`` where **t is late-positive** (``t = c·Δt``) and
**pt is the negative energy deviation** (``RefPart.pt = −γ``).  Both differ in sign from
MAD-X's ``(T, pt)``, and the two flips cancel in ``R55``, ``R56``, ``R65``, ``R66`` — a
1 m drift gives ``R56 = +223.148 395 685 4 = +L/(β²γ²)`` in *both* codes — but **not** in
the dispersion column (``R16…R46``) or the path-length row (``R51…R54``), which come out
with the opposite sign to MAD-X.

This adapter therefore returns maps conjugated with ``S = diag(1,1,1,1,−1,−1)``, i.e. in
MAD-X's ``(T, pt)`` convention, which is what :data:`lattix.oracles.basis._Z_SIGN`
already assumes for :attr:`Basis.IMPACTX` (``+1`` with ``δ = pt/β``).  ``meta`` records
the raw ImpactX numbers.  With that conjugation the common-basis drift map reproduces
:func:`lattix.oracles.basis.drift_common` to 6e-16 and the whole ``fodo.madx`` cumulative
map agrees with cpymad to 1.8e-15.

Reference gain: ``ShortRF(V, freq, phase)`` has ``V = ΔE_max/mc²`` and ``phase`` in
degrees with **0 = crest**, identical to the IR's synchronous phase.  MEASURED: a proton
at 2.1 MeV through ``ShortRF(V = 1 MV/mc², 162.5 MHz, phase = −30°)`` gains
866 025.403 784 4 eV = +V·cos 30°, and the map has ``R22 = R66 = p_in/p_out`` and
``R65_common = −4.3046`` (bunching), matching Bmad/HELIX (−4.305) and TraceWin (−4.273).

Known ImpactX bug on this machine: ``ImpactX.load_inputs_file()`` (``ParmParse::addfile``)
leaves AMReX in a state where the next call segfaults — reproduced with ImpactX's own
``examples/fodo/input_fodo.in`` and ``tests/python/test_impactx.py::test_impactx_fodo_file``
(exit 139), even with an empty file.  The adapter therefore never uses it: an ``impactx``
inputs deck is read by :mod:`lattix.formats.impactx` and rebuilt through the Python API,
and a ``.madx`` deck goes through :mod:`lattix.formats.madx` + the ImpactX writer (or,
with ``native_madx=True``, through ImpactX's own ``KnownElementsList.load_file``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np

#: sign flip between ImpactX's (t, pt) and MAD-X's (T, pt)
_S = np.diag([1.0, 1.0, 1.0, 1.0, -1.0, -1.0])

#: probe half-amplitude for the central differences (ImpactX maps are linear, so this is
#: only limited by round-off; 1e-7 keeps 9 significant digits on a 1e-16 double)
DEFAULT_H = 1e-7


# ===========================================================================
# worker half — imports only impactx, numpy and the standard library
# ===========================================================================
def _pod(amr, values):
    """pyAMReX wants a ``PODVector_real_std``; older/newer builds accept numpy."""
    try:
        v = amr.PODVector_real_std()
    except Exception:  # pragma: no cover - depends on the pyAMReX build
        return np.asarray(values, dtype=float)
    for x in np.asarray(values, dtype=float):
        v.push_back(float(x))
    return v


def _build_element(elements, amr, spec: dict):
    cls = getattr(elements, spec["cls"])
    params = dict(spec.get("params", {}))
    if "R" in params and isinstance(params["R"], list):
        m = amr.SmallMatrix_6x6_F_SI1_double()
        base = getattr(m, "starting_index", 1)
        for i, row in enumerate(params["R"], start=base):
            for j, v in enumerate(row, start=base):
                try:
                    m[i, j] = float(v)
                except TypeError:  # pragma: no cover - other pyAMReX builds
                    m.set_val(i, j, float(v))
        params["R"] = m
    name = spec.get("name")
    if spec["cls"] in ("Marker", "BeamMonitor"):
        return cls(name=name, **{k: v for k, v in params.items() if k != "name"})
    if name is not None:
        params["name"] = name
    return cls(**params)


def _read_coords(pc):
    df = pc.to_df(local=True)
    if df is None or len(df) == 0:  # pragma: no cover - only when every probe is lost
        return np.zeros((0, 6))
    a = np.stack([df["position_x"].to_numpy(), df["momentum_x"].to_numpy(),
                  df["position_y"].to_numpy(), df["momentum_y"].to_numpy(),
                  df["position_t"].to_numpy(), df["momentum_t"].to_numpy()], axis=1)
    return a[np.argsort(df["idcpu"].to_numpy())]


def run_worker(spec: dict) -> dict:
    """Track the probe through every element; return the maps and reference energies.

    ``spec``: ``{"elements": [{"cls", "name", "params"}], "mass_MeV", "charge_qe",
    "kin_energy_MeV", "h", "probe": [[…6…]] | None, "madx_file": str | None,
    "nslice": int}``.  All matrices come back in **raw ImpactX** ``(t, pt)``.
    """
    import amrex.space3d as amr
    import impactx
    from impactx import ImpactX, elements

    # ImpactX's init_grids() creates (and rotates) a `diags/` directory in the CWD even
    # with diagnostics off, so the whole run happens inside the caller's workdir.
    cwd = os.getcwd()
    if spec.get("workdir"):
        Path(spec["workdir"]).mkdir(parents=True, exist_ok=True)
        os.chdir(spec["workdir"])
    try:
        return _run_lattice(spec, amr, impactx, ImpactX, elements)
    finally:
        os.chdir(cwd)


def _run_lattice(spec, amr, impactx, ImpactX, elements) -> dict:  # noqa: N803
    mass_MeV = float(spec["mass_MeV"])
    charge_qe = float(spec["charge_qe"])
    ke_MeV = float(spec["kin_energy_MeV"])
    h = float(spec.get("h", DEFAULT_H))
    warnings: list[str] = []

    sim = ImpactX()
    sim.verbose = 0
    sim.tiny_profiler = False
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.init_grids()
    pc = sim.particle_container()
    ref = pc.ref_particle()
    ref.set_charge_qe(charge_qe)
    ref.set_mass_MeV(mass_MeV)
    ref.set_kin_energy_MeV(ke_MeV)

    if spec.get("madx_file"):
        kl = elements.KnownElementsList()
        try:
            kl.load_file(spec["madx_file"], nslice=int(spec.get("nslice", 1)))
        except TypeError:
            kl.load_file(spec["madx_file"])
        lattice = list(kl)
    else:
        lattice = [_build_element(elements, amr, s) for s in spec["elements"]]

    qm = charge_qe / (mass_MeV * 1.0e6)
    seed = np.zeros((13, 6))
    for i in range(6):
        seed[1 + 2 * i, i] = +h
        seed[2 + 2 * i, i] = -h

    def reseed(rows):
        pc.clear_particles()
        order = (0, 2, 4, 1, 3, 5)          # x, y, t, px, py, pt
        try:
            pc.add_n_particles(*[_pod(amr, rows[:, j]) for j in order], qm, 1.0e-12)
        except TypeError:  # pragma: no cover - builds that want plain arrays
            pc.add_n_particles(*[np.ascontiguousarray(rows[:, j]) for j in order],
                               qm, 1.0e-12)

    names, types, lengths, s_out, R_all, e_in, e_out = [], [], [], [], [], [], []
    s = 0.0
    for el in lattice:
        reseed(seed)
        before = _read_coords(pc)
        ke0 = ref.kin_energy_MeV
        for _ in range(max(1, int(getattr(el, "nslice", 1) or 1))):
            el.push(pc, 0, 0)
        after = _read_coords(pc)
        ke1 = ref.kin_energy_MeV
        R = np.zeros((6, 6))
        if len(after) == len(before) == 13:
            for i in range(6):
                den = before[1 + 2 * i, i] - before[2 + 2 * i, i]
                R[:, i] = (after[1 + 2 * i] - after[2 + 2 * i]) / den
        else:  # pragma: no cover - a probe was absorbed by an aperture
            R = np.eye(6)
            warnings.append(f"probe lost in {getattr(el, 'name', '?')}; identity map assumed")
        ds = float(getattr(el, "ds", 0.0) or 0.0)
        s += ds
        names.append(getattr(el, "name", None) or f"e{len(names)}")
        types.append(type(el).__name__)
        lengths.append(ds)
        s_out.append(s)
        R_all.append(R.tolist())
        e_in.append(ke0)
        e_out.append(ke1)

    probe_out = None
    if spec.get("probe"):
        probe = np.asarray(spec["probe"], dtype=float)
        ref.set_kin_energy_MeV(ke_MeV)
        reseed(probe)
        for el in lattice:
            for _ in range(max(1, int(getattr(el, "nslice", 1) or 1))):
                el.push(pc, 0, 0)
        probe_out = _read_coords(pc).tolist()

    return {
        "names": names, "types": types, "length": lengths, "s_out": s_out,
        "R": R_all, "kin_in_MeV": e_in, "kin_out_MeV": e_out,
        "mass_MeV": float(ref.mass_MeV), "charge_qe": float(ref.charge_qe),
        "probe_out": probe_out, "warnings": warnings,
        "impactx_version": getattr(impactx, "__version__", "?"),
        "numpy_version": np.__version__,
    }


def _worker_main(argv: list[str]) -> int:
    spec = json.loads(Path(argv[0]).read_text())
    data = run_worker(spec)
    Path(argv[1]).write_text(json.dumps(data))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the subprocess path
    raise SystemExit(_worker_main(sys.argv[1:]))


# ===========================================================================
# adapter half
# ===========================================================================
from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register  # noqa: E402

_CHECK = "import impactx, sys; print(sys.executable)"
_WORKER = Path(__file__).resolve()


def _can_import(python: str, timeout: float = 300.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run([python, "-I", "-c", _CHECK], capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{python}: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"{python}: {tail[-1] if tail else 'exit ' + str(proc.returncode)}"
    lines = proc.stdout.strip().splitlines()
    return (True, lines[-1]) if lines else (False, f"{python}: no output")


def _conda_roots() -> list[Path]:
    roots: list[Path] = []
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        p = Path(prefix)
        roots += [p, p.parent.parent if p.parent.name == "envs" else p.parent]
    exe = os.environ.get("CONDA_EXE")
    if exe:
        roots.append(Path(exe).resolve().parent.parent)
    home = Path.home()
    roots += [home / d for d in ("anaconda3", "miniconda3", "miniforge3", "mambaforge", "conda")]
    roots += [Path(p) for p in ("/opt/conda", "/opt/anaconda3", "/opt/miniconda3",
                                "/usr/local/anaconda3", "/usr/local/miniconda3")]
    seen, out = set(), []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _resolve_python(env: str) -> tuple[str | None, list[str]]:
    tried: list[str] = []
    explicit = os.environ.get("LATTIX_IMPACTX_PYTHON")
    if explicit:
        ok, why = _can_import(explicit)
        if ok:
            return why, tried
        tried.append(f"LATTIX_IMPACTX_PYTHON {why}")
    exe = "python.exe" if sys.platform == "win32" else "bin/python"
    for root in _conda_roots():
        cand = root / "envs" / env / exe
        if cand.exists():
            ok, why = _can_import(str(cand))
            if ok:
                return why, tried
            tried.append(why)
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        try:
            proc = subprocess.run([conda, "run", "-n", env, "python", "-c", _CHECK],
                                  capture_output=True, text=True, timeout=600, check=False)
            lines = proc.stdout.strip().splitlines()
            if proc.returncode == 0 and lines and Path(lines[-1]).exists():
                return lines[-1], tried
            err = (proc.stderr or proc.stdout).strip().splitlines()
            tried.append(f"conda run -n {env}: {err[-1] if err else 'exit ' + str(proc.returncode)}")
        except (OSError, subprocess.TimeoutExpired) as e:
            tried.append(f"conda run -n {env}: {e}")
    else:
        tried.append("no conda executable on PATH / CONDA_EXE")
    return None, tried


@register
class ImpactxOracle:
    """ImpactX 26.x adapter.  ``formats`` are the deck formats it consumes directly."""

    name = "impactx"
    formats = ("impactx", "madx")

    #: (ok, reason, python path or None, in-process?) — resolved once per process
    _resolved: ClassVar[tuple[bool, str, str | None, bool] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    @classmethod
    def _resolve(cls) -> tuple[bool, str, str | None, bool]:
        if cls._resolved is None:
            try:
                import impactx  # noqa: F401

                cls._resolved = (True, "impactx in-process", sys.executable, True)
            except Exception:  # noqa: BLE001
                env = os.environ.get("LATTIX_IMPACTX_ENV", "lattix")
                python, tried = _resolve_python(env)
                if python:
                    cls._resolved = (True, f"impactx via {python}", python, False)
                else:
                    cls._resolved = (False, "no interpreter with impactx found (set "
                                            "LATTIX_IMPACTX_PYTHON or LATTIX_IMPACTX_ENV; tried: "
                                            f"{'; '.join(tried) or 'nothing'})", None, False)
        return cls._resolved

    def available(self) -> tuple[bool, str]:
        ok, why, _, _ = self._resolve()
        return ok, why

    # ------------------------------------------------------------------
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None, nslice: int = 1,
            native_madx: bool = False, h: float = DEFAULT_H,
            instrument: str = "marker") -> OracleResult:
        from lattix.oracles.basis import transform_matrix

        ok, why, python, inprocess = self._resolve()
        if not ok:
            raise RuntimeError(f"impactx oracle unavailable: {why}")
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        fmt = fmt or ("madx" if deck.suffix.lower() in (".madx", ".seq", ".mad") else "impactx")
        if fmt not in self.formats:
            raise ValueError(f"impactx oracle reads {self.formats}, not {fmt!r}")
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_impactx_"))
        wd.mkdir(parents=True, exist_ok=True)

        spec, lat_report = self._spec(deck, fmt, beam, nslice=nslice, native_madx=native_madx,
                                      instrument=instrument)
        spec["h"] = float(h)
        spec["workdir"] = str(wd)
        mass_eV = spec["mass_MeV"] * 1e6
        if probe is not None:
            T = transform_matrix(Basis.IMPACTX, spec["kin_energy_MeV"] * 1e6, mass_eV)
            native = np.asarray(probe.coords, dtype=float) @ np.linalg.inv(T).T
            spec["probe"] = (native @ _S).tolist()          # MAD-X-like -> raw ImpactX

        data = run_worker(spec) if inprocess else self._run_subprocess(python, spec, wd)
        return self._to_result(data, spec, wd, python, inprocess, lat_report, fmt, deck)

    # ------------------------------------------------------------------
    @staticmethod
    def _spec(deck: Path, fmt: str, beam: BeamSpec | None, *, nslice: int, native_madx: bool,
              instrument: str) -> tuple[dict, object]:
        """Deck -> the worker's JSON spec (via the lattix reader + ImpactX writer)."""
        from lattix.formats.impactx.writer import to_elements

        opt = beam or BeamSpec()
        if fmt == "madx" and native_madx:
            if beam is None:
                raise ValueError("native_madx=True needs an explicit beam: ImpactX's own MAD-X "
                                 "importer returns only the element list, not a reference particle")
            return ({"elements": [], "madx_file": str(deck), "nslice": int(nslice),
                     "mass_MeV": opt.mass_eV * 1e-6, "charge_qe": float(opt.charge),
                     "kin_energy_MeV": opt.kinetic_energy_eV * 1e-6}, None)

        if fmt == "madx":
            from lattix.formats.madx import Reader as SrcReader
        else:
            from lattix.formats.impactx import Reader as SrcReader
        lat, rep_in = SrcReader().read(deck)
        rep_in.target_format = "impactx"
        emits, rep_out = to_elements(lat, flavor="python", nslice=nslice, instrument=instrument)
        rep_in.extend(rep_out)
        # The deck's own reference converts its strengths (so an ImpactX ``k`` survives a
        # read/write round trip bit for bit, and a MAD-X ``k1`` passes through unchanged);
        # ``beam`` only retargets the reference particle ImpactX runs with — the same
        # "normalized strengths stay, energy moves" semantics cpymad and the Bmad worker
        # give when a BeamSpec overrides a deck's own energy.
        ref = lat.reference
        return ({"elements": [{"cls": e.cls, "name": e.name, "params": e.params} for e in emits],
                 "madx_file": None, "nslice": int(nslice),
                 "mass_MeV": (opt.mass_eV if beam is not None else ref.species.mass_eV) * 1e-6,
                 "charge_qe": float(opt.charge if beam is not None else ref.species.charge),
                 "kin_energy_MeV": (opt.kinetic_energy_eV if beam is not None
                                    else ref.kinetic_energy_eV) * 1e-6}, rep_in)

    @staticmethod
    def _run_subprocess(python: str, spec: dict, wd: Path) -> dict:
        spec_path, out_path = wd / "impactx_spec.json", wd / "impactx_result.json"
        spec_path.write_text(json.dumps(spec))
        cmd = [python, "-I", str(_WORKER), str(spec_path), str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wd), check=False)
        if proc.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"impactx worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n"
                               f"{proc.stdout[-2000:]}")
        return json.loads(out_path.read_text())

    @staticmethod
    def _to_result(data: dict, spec: dict, wd: Path, python: str | None, inprocess: bool,
                   lat_report, fmt: str, deck: Path) -> OracleResult:
        n = len(data["names"])
        mass = float(data["mass_MeV"]) * 1e6
        raw = np.asarray(data["R"], dtype=float).reshape(n, 6, 6) if n else np.zeros((0, 6, 6))
        R = np.array([_S @ m @ _S for m in raw]) if n else raw
        probe_out = data.get("probe_out")
        if probe_out is not None:
            probe_out = np.asarray(probe_out, dtype=float) @ _S
        meta = {
            "python": python, "mode": "in-process" if inprocess else "subprocess",
            "impactx_version": data.get("impactx_version"),
            "numpy_version": data.get("numpy_version"),
            "source_format": fmt, "deck": str(deck), "workdir": str(wd),
            "element_types": list(data.get("types", [])),
            "sign_convention": "raw ImpactX (t, pt) is (late-positive, -dE/p0c); the adapter "
                               "returns S·R·S with S = diag(1,1,1,1,-1,-1), i.e. MAD-X's (T, pt)",
            "R_raw_impactx": raw.tolist(),
            "p0_model": "follows p0: ShortRF pushes the reference particle; pt is relative to "
                        "the local p0",
            "fidelity": None if lat_report is None else lat_report.model_dump(mode="json"),
        }
        return OracleResult(
            engine="impactx", basis=Basis.IMPACTX, names=[str(x) for x in data["names"]],
            length=np.asarray(data["length"], dtype=float),
            s_out=np.asarray(data["s_out"], dtype=float), R_elem=R,
            ref_kinetic_eV_in=np.asarray(data["kin_in_MeV"], dtype=float) * 1e6,
            ref_kinetic_eV_out=np.asarray(data["kin_out_MeV"], dtype=float) * 1e6,
            mass_eV=mass, charge=int(round(float(data["charge_qe"]))),
            probe_out=probe_out, warnings=[str(w) for w in data.get("warnings", [])], meta=meta,
        )
