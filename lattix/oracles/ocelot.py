"""Ocelot oracle: per-element first-order maps of an Ocelot lattice module, evaluated by Ocelot 25.06 in
its own environment (GPL-3, never imported here) through :mod:`lattix.oracles.ocelot_worker`.

Measured on Ocelot 25.06.0 (2026-09-05, docs/oracles.md Phase 5.7): maps in MAD-X's basis
``(x, px, y, py, τ late-positive [m], ΔE/(p0 c))`` — a 1 m drift gives ``R56 = −L/(β²γ²)``; the reference
energy follows the cavities (``Cavity`` gains ``v·cos(phi)`` GeV; ``R65 ∝ +sin(phi)``, so the IR's
bunching phase is ``phi = −φs``); every map divides by the electron mass (an electron or positron beam is
the only faithful one — other species are report only).  The interpreter is found through
``LATTIX_OCELOT_PYTHON``, the current interpreter, the environment ``ocelot`` (``LATTIX_OCELOT_ENV``:
``<conda root>/envs/ocelot/bin/python``, a venv there works too) or ``conda run``.  An Elegant ``.lte``
deck is read by Ocelot's own ``ElegantLatticeConverter`` (``fmt="elegant"``) for lockstep checks.
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

_WORKER = Path(__file__).with_name("ocelot_worker.py")


@register
class OcelotOracle:
    name = "ocelot"
    formats = ("ocelot", "elegant")

    _resolved: ClassVar[tuple[bool, str, str | None] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    def available(self) -> tuple[bool, str]:
        if OcelotOracle._resolved is None:
            env = os.environ.get("LATTIX_OCELOT_ENV", "ocelot")
            python, tried = resolve_python("ocelot", env, "LATTIX_OCELOT_PYTHON")
            if python:
                OcelotOracle._resolved = (True, f"Ocelot via {python}", python)
            else:
                OcelotOracle._resolved = (False, "no interpreter imports ocelot: " + "; ".join(tried), None)
        ok, why, _ = OcelotOracle._resolved
        return ok, why

    def _python(self) -> str:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"ocelot oracle unavailable: {why}")
        return OcelotOracle._resolved[2]

    def _run_worker(self, deck: Path, out: Path, args: list[str], cwd: Path) -> dict:
        cmd = [self._python(), "-I", str(_WORKER), str(deck), "--out", str(out), *args]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ocelot worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n{proc.stdout[-2000:]}")
        return json.loads(out.read_text())

    def dump(self, deck: Path, workdir: Path | None = None) -> Path:
        """Run the module in the Ocelot environment and return the JSON file with its sequence."""
        deck = Path(deck).resolve()
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_ocelot_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "ocelot_dump.json"
        self._run_worker(deck, out, ["--dump"], wd)
        return out

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None) -> OracleResult:
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        if beam is None:
            from lattix.ir.reference_tag import parse_reference_tag

            tag = parse_reference_tag(deck.read_text(encoding="utf-8", errors="replace"))
            if tag is None:
                raise ValueError("the ocelot oracle needs a BeamSpec or a lattix reference tag in the file")
            beam = BeamSpec(species=tag.species.name, kinetic_energy_eV=tag.kinetic_energy_eV,
                            frequency_Hz=tag.rf_frequency_Hz)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_ocelot_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "ocelot_result.json"
        args = ["--ke-ev", repr(float(beam.kinetic_energy_eV)), "--mass-ev", repr(float(beam.mass_eV)),
                "--charge", repr(float(beam.charge)), "--species", str(beam.species)]
        if fmt == "elegant" or (fmt is None and deck.suffix.lower() == ".lte"):
            args.append("--elegant")
        data = self._run_worker(deck, out, args, wd)
        names = [str(x) for x in data["names"]]
        n = len(names)
        R = np.asarray(data["R"], dtype=float).reshape(n, 6, 6)
        meta = {"workdir": str(wd), "ocelot_version": data.get("ocelot_version"), "python": data.get("python"),
                "kinds": list(data.get("kinds", [])), "p0_model": data.get("p0_model"),
                "electron_only": beam.species.lower() not in ("electron", "positron")}
        return OracleResult(engine="ocelot", basis=Basis.OCELOT, names=names,
                            length=np.asarray(data["length"], dtype=float),
                            s_out=np.asarray(data["s_out"], dtype=float), R_elem=R,
                            ref_kinetic_eV_in=np.asarray(data["w_in_ev"], dtype=float),
                            ref_kinetic_eV_out=np.asarray(data["w_out_ev"], dtype=float),
                            mass_eV=float(beam.mass_eV), charge=int(beam.charge),
                            warnings=[str(w) for w in data.get("warnings", [])], meta=meta)
