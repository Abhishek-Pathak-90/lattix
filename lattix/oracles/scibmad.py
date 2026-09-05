"""SciBmad oracle: BeamTracking.jl through a Julia subprocess (:file:`scibmad_worker.jl`).

Measured on SciBmad 0.5.2 / Julia 1.10 (2026-09-05, docs/oracles.md): native basis is Bmad's
``(x, px, y, py, z, pz)`` with ``z`` ahead-positive (a 1 m drift at 2.1 MeV gives ``R56 = +L/γ²``),
one reference momentum per beamline (a cavity's gain goes into ``pz``: constant-p0 engine), the RF
gain is ``−V·cos(phi0)``, zero-length cavities and fringe integrals cannot be tracked (the worker
substitutes and reports).  Per-element maps are central finite differences around the tracked
reference orbit, like the xtrack oracle.  ``julia`` is found through ``LATTIX_JULIA`` or PATH;
the availability probe (``using SciBmad``) takes ~15 s and is cached per process.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register

_WORKER = Path(__file__).with_name("scibmad_worker.jl")
_CHECK = 'using SciBmad; print(string(pkgversion(SciBmad)))'


def julia_executable() -> str | None:
    env = os.environ.get("LATTIX_JULIA")
    if env and Path(env).exists():
        return env
    return shutil.which("julia")


@register
class ScibmadOracle:
    name = "scibmad"
    formats = ("scibmad",)
    _cache: tuple[bool, str] | None = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._cache = None

    def available(self) -> tuple[bool, str]:
        if ScibmadOracle._cache is not None:
            return ScibmadOracle._cache
        julia = julia_executable()
        if not julia:
            ScibmadOracle._cache = (False, "julia not found (set LATTIX_JULIA or add julia to PATH)")
            return ScibmadOracle._cache
        try:
            proc = subprocess.run([julia, "--startup-file=no", "-e", _CHECK], capture_output=True, text=True,
                                  timeout=600, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            ScibmadOracle._cache = (False, f"julia failed: {e}")
            return ScibmadOracle._cache
        if proc.returncode != 0:
            ScibmadOracle._cache = (False, f"`using SciBmad` failed: {proc.stderr.strip()[-200:]}")
        else:
            ScibmadOracle._cache = (True, f"SciBmad {proc.stdout.strip()} via {julia}")
        return ScibmadOracle._cache

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            line: str | None = None) -> OracleResult:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"scibmad oracle unavailable: {why}")
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_scibmad_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "scibmad_result.json"
        cmd = [julia_executable(), "--startup-file=no", str(_WORKER), str(deck), "--out", str(out)]
        if line:
            cmd += ["--line", line]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wd), check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"scibmad worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n{proc.stdout[-2000:]}")
        data = json.loads(out.read_text())
        return self._to_result(data, wd, beam)

    @staticmethod
    def _to_result(data: dict, wd: Path, beam: BeamSpec | None) -> OracleResult:
        names = [str(x) for x in data["names"]]
        n = len(names)
        mass = float(data["mass_ev"])
        charge = int(round(float(data["charge"])))
        R = np.asarray(data["mat6"], dtype=float).reshape(n, 6, 6)
        e_in = np.asarray(data["e_tot_in"], dtype=float) - mass
        e_out = np.asarray(data["e_tot_out"], dtype=float) - mass
        warnings = [str(w) for w in data.get("warnings", [])]
        if beam is not None and abs(beam.kinetic_energy_eV - e_in[0]) > 1e-6 * max(1.0, abs(e_in[0])):
            warnings.append(f"the deck's reference energy ({e_in[0]:.6g} eV) is used, not the requested "
                            f"{beam.kinetic_energy_eV:.6g} eV")
        meta = {
            "workdir": str(wd), "scibmad_version": data.get("scibmad_version"),
            "julia_version": data.get("julia_version"), "species": data.get("species"),
            "root": data.get("root"), "kinds": list(data.get("kinds", [])),
            "p0_model": data.get("p0_model"), "fd": "central differences around the tracked reference orbit",
            "p_over_q_ref": data.get("p_over_q_ref"),
        }
        return OracleResult(engine="scibmad", basis=Basis.BMAD, names=names,
                            length=np.asarray(data["length"], dtype=float),
                            s_out=np.asarray(data["s_out"], dtype=float), R_elem=R,
                            ref_kinetic_eV_in=e_in, ref_kinetic_eV_out=e_out, mass_eV=mass, charge=charge,
                            warnings=warnings, meta=meta)
