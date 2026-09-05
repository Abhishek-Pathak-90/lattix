"""Cheetah oracle: per-element first-order maps of a LatticeJSON lattice, evaluated by Cheetah 0.8 in its
own environment (torch; GPL-3, never imported here) through :mod:`lattix.oracles.cheetah_worker`.

Measured on Cheetah 0.8.4 (2026-09-05, docs/oracles.md): maps in MAD-X's basis
``(x, px, y, py, τ late-positive [m], ΔE/(p0 c))`` — a 1 m drift at 2.1 MeV gives
``R56 = −L/(β²γ²)``; the reference energy follows the cavities (``ΔE = −voltage·q·cos(phase)``);
``k1``/``k`` are charge-blind (the writer normalizes with the signed rigidity).  The interpreter is
found through ``LATTIX_CHEETAH_PYTHON``, the current interpreter, the conda env ``cheetah``
(``LATTIX_CHEETAH_ENV``) or ``conda run`` (:mod:`lattix.oracles.envs`).
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

_WORKER = Path(__file__).with_name("cheetah_worker.py")


@register
class CheetahOracle:
    name = "cheetah"
    formats = ("cheetah", "bmad")           # a .bmad deck goes through Cheetah's own Bmad converter

    _resolved: ClassVar[tuple[bool, str, str | None] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    def available(self) -> tuple[bool, str]:
        if CheetahOracle._resolved is None:
            env = os.environ.get("LATTIX_CHEETAH_ENV", "cheetah")
            python, tried = resolve_python("cheetah", env, "LATTIX_CHEETAH_PYTHON")
            if python:
                CheetahOracle._resolved = (True, f"Cheetah via {python}", python)
            else:
                CheetahOracle._resolved = (False, "no interpreter imports cheetah: " + "; ".join(tried), None)
        ok, why, _ = CheetahOracle._resolved
        return ok, why

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None) -> OracleResult:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"cheetah oracle unavailable: {why}")
        python = CheetahOracle._resolved[2]
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        if beam is None:
            from lattix.ir.reference_tag import parse_reference_tag

            tag = parse_reference_tag(json.loads(deck.read_text()).get("info") or "")
            if tag is None:
                raise ValueError("the cheetah oracle needs a BeamSpec or a lattix reference tag in the file's info")
            beam = BeamSpec(species=tag.species.name, kinetic_energy_eV=tag.kinetic_energy_eV,
                            frequency_Hz=tag.rf_frequency_Hz)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_cheetah_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "cheetah_result.json"
        cmd = [python, "-I", str(_WORKER), str(deck), "--out", str(out),
               "--ke-ev", repr(float(beam.kinetic_energy_eV)), "--mass-ev", repr(float(beam.mass_eV)),
               "--charge", repr(float(beam.charge)), "--species", str(beam.species)]
        if fmt == "bmad" or (fmt is None and deck.suffix == ".bmad"):
            cmd.append("--bmad")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wd), check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"cheetah worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n{proc.stdout[-2000:]}")
        data = json.loads(out.read_text())
        names = [str(x) for x in data["names"]]
        n = len(names)
        R = np.asarray(data["R"], dtype=float).reshape(n, 6, 6)
        meta = {"workdir": str(wd), "cheetah_version": data.get("cheetah_version"),
                "torch_version": data.get("torch_version"), "python": data.get("python"),
                "kinds": list(data.get("kinds", [])), "p0_model": data.get("p0_model")}
        return OracleResult(engine="cheetah", basis=Basis.CHEETAH, names=names,
                            length=np.asarray(data["length"], dtype=float),
                            s_out=np.asarray(data["s_out"], dtype=float), R_elem=R,
                            ref_kinetic_eV_in=np.asarray(data["w_in_ev"], dtype=float),
                            ref_kinetic_eV_out=np.asarray(data["w_out_ev"], dtype=float),
                            mass_eV=float(beam.mass_eV), charge=int(beam.charge),
                            warnings=[str(w) for w in data.get("warnings", [])], meta=meta)
