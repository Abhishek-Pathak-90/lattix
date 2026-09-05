"""PyORBIT3 oracle: per-element transport matrices of a linac XML lattice from PyORBIT3's
``LinacTrMatricesController`` in its own environment (:mod:`lattix.oracles.pyorbit_worker`): symmetric
probes tracked at two amplitudes and Richardson-extrapolated (``probe_scale`` rescales them).

Measured on PyORBIT3 (meson build, 2026-09-05, docs/oracles.md): coordinates
``(x [m], x', y [m], y', z [m] ahead-positive, dE [GeV])`` — a 1 m drift at 2.1 MeV gives
``R56 = +237.30 = L/γ² · 10⁹/(β²γ mc²)`` (``Basis.PYORBIT``); the reference energy follows the gaps
(``ΔE = q·E0TL·cos(phase)``); quadrupole ``field`` is the lab gradient with the charge in the
rigidity, corrector ``B·effLength`` kicks scale with the signed rigidity, ``SOLENOID B`` is the
normalized strength ``B₀/Bρ`` in 1/m.  The interpreter is found through ``LATTIX_PYORBIT_PYTHON``,
the current interpreter, the conda env ``pyorbit`` (``LATTIX_PYORBIT_ENV``) or ``conda run``.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np

from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register
from lattix.oracles.envs import resolve_python

_WORKER = Path(__file__).with_name("pyorbit_worker.py")


@register
class PyorbitOracle:
    name = "pyorbit"
    formats = ("pyorbit",)

    _resolved: ClassVar[tuple[bool, str, str | None] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    def available(self) -> tuple[bool, str]:
        if PyorbitOracle._resolved is None:
            env = os.environ.get("LATTIX_PYORBIT_ENV", "pyorbit")
            # ``orbit`` alone: ``orbit.core`` needs libfftw3 preloaded on macOS (the worker does that)
            python, tried = resolve_python("orbit", env, "LATTIX_PYORBIT_PYTHON")
            if python:
                PyorbitOracle._resolved = (True, f"PyORBIT3 via {python}", python)
            else:
                PyorbitOracle._resolved = (False, "no interpreter imports orbit: " + "; ".join(tried), None)
        ok, why, _ = PyorbitOracle._resolved
        return ok, why

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            gap_model: str = "BaseRfGap", probe_scale: float = 1.0) -> OracleResult:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"pyorbit oracle unavailable: {why}")
        python = PyorbitOracle._resolved[2]
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        if beam is None:
            from lattix.ir.reference_tag import parse_reference_tag

            tag = parse_reference_tag(deck.read_text(encoding="utf-8", errors="replace"))
            if tag is None:
                raise ValueError("the pyorbit oracle needs a BeamSpec or a lattix reference tag in the XML")
            beam = BeamSpec(species=tag.species.name, kinetic_energy_eV=tag.kinetic_energy_eV,
                            frequency_Hz=tag.rf_frequency_Hz)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_pyorbit_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "pyorbit_result.json"
        cmd = [python, "-I", str(_WORKER), str(deck), "--out", str(out),
               "--ke-ev", repr(float(beam.kinetic_energy_eV)), "--mass-ev", repr(float(beam.mass_eV)),
               "--probe-scale", repr(float(probe_scale)),
               "--charge", repr(float(beam.charge)), "--gap-model", gap_model]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wd), check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"pyorbit worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n{proc.stdout[-2000:]}")
        data = json.loads(out.read_text())
        names = [str(x) for x in data["names"]]
        n = len(names)
        R = np.asarray(data["R"], dtype=float).reshape(n, 6, 6)
        meta = {"workdir": str(wd), "python": data.get("python"), "gap_model": data.get("gap_model"),
                "sequences": data.get("sequences"), "kinds": list(data.get("kinds", [])),
                "lattice_length": data.get("lattice_length"), "p0_model": data.get("p0_model"),
                "fftw_preloaded": data.get("fftw_preloaded")}
        return OracleResult(engine="pyorbit", basis=Basis.PYORBIT, names=names,
                            length=np.asarray(data["length"], dtype=float),
                            s_out=np.asarray(data["s_out"], dtype=float), R_elem=R,
                            ref_kinetic_eV_in=np.asarray(data["w_in_ev"], dtype=float),
                            ref_kinetic_eV_out=np.asarray(data["w_out_ev"], dtype=float),
                            mass_eV=float(beam.mass_eV), charge=int(beam.charge),
                            warnings=[str(w) for w in data.get("warnings", [])], meta=meta)
