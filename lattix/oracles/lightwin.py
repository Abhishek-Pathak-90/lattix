"""LightWin oracle: the ``Envelope3D`` beam calculator of LightWin (TraceWin semantics, MIT) run
through :mod:`lattix.oracles.lightwin_worker` in LightWin's own environment (python ≥ 3.12).

Measured on LightWin 0.16.5 (2026-09-05, docs/oracles.md): native basis is TraceWin's
``(x, x', y, y', z [m], dp/p)`` with ``z`` ahead-positive (a 1 m drift at 2.1 MeV gives
``R56 = +L/γ² = 0.9955387``, the xtrack/Bmad number), quadrupoles focus ``x`` for ``q·G > 0``
(TraceWin's rule), the reference energy follows the field maps, and every element without a 3-D
model (GAP, EDGE, THIN_STEERING, …) is a drift — the adapter lists them in ``meta["substituted"]``
and warns.  Field-map files are resolved by LightWin relative to the deck, like TraceWin.

The interpreter is found through ``LATTIX_LIGHTWIN_PYTHON``, the current interpreter, the conda env
``lightwin`` (``LATTIX_LIGHTWIN_ENV``) or ``conda run``; see :mod:`lattix.oracles.envs`.
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

_WORKER = Path(__file__).with_name("lightwin_worker.py")


@register
class LightwinOracle:
    name = "lightwin"
    formats = ("tracewin",)

    #: (ok, reason, python path or None) — resolved once per process
    _resolved: ClassVar[tuple[bool, str, str | None] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    def available(self) -> tuple[bool, str]:
        if LightwinOracle._resolved is None:
            env = os.environ.get("LATTIX_LIGHTWIN_ENV", "lightwin")
            python, tried = resolve_python("lightwin", env, "LATTIX_LIGHTWIN_PYTHON")
            if python:
                LightwinOracle._resolved = (True, f"LightWin via {python}", python)
            else:
                LightwinOracle._resolved = (False, "no interpreter imports lightwin: " + "; ".join(tried), None)
        ok, why, _ = LightwinOracle._resolved
        return ok, why

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            n_steps_per_cell: int = 40, phase_policy: str = "as_in_original_dat",
            parse_only: bool = False) -> OracleResult | dict:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"lightwin oracle unavailable: {why}")
        python = LightwinOracle._resolved[2]
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        if beam is None:
            raise ValueError("the lightwin oracle needs a BeamSpec (species, kinetic energy, bunch frequency)")
        mass, charge = beam.mass_eV, beam.charge
        if not beam.frequency_Hz:
            raise ValueError("BeamSpec.frequency_Hz (the bunch frequency) is required for LightWin")
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_lightwin_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "lightwin_result.json"
        cmd = [python, "-I", str(_WORKER), str(deck), "--out", str(out),
               "--ke-ev", repr(float(beam.kinetic_energy_eV)), "--mass-ev", repr(float(mass)),
               "--charge", repr(float(charge)), "--freq-hz", repr(float(beam.frequency_Hz)),
               "--n-steps", str(int(n_steps_per_cell)), "--phase-policy", phase_policy]
        if parse_only:
            cmd.append("--parse-only")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wd), check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"lightwin worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n{proc.stdout[-2000:]}")
        data = json.loads(out.read_text())
        if parse_only:
            return data
        return self._to_result(data, wd, beam, mass, charge)

    @staticmethod
    def _to_result(data: dict, wd: Path, beam: BeamSpec, mass: float, charge: int) -> OracleResult:
        names = [str(x) for x in data["names"]]
        n = len(names)
        R = np.asarray(data["R"], dtype=float).reshape(n, 6, 6)
        warnings = [str(w) for w in data.get("warnings", [])]
        subs = list(data.get("substituted", []))
        from lattix.formats.tracewin.reader import _is_command

        dummies = list(data.get("dropped", []))
        dropped = [d for d in dummies if not _is_command(str(d["keyword"]))]
        ignored = [d for d in dummies if _is_command(str(d["keyword"]))]
        if dropped:
            kws = sorted({d["keyword"] for d in dropped})
            warnings.append(f"LightWin has no element for {len(dropped)} deck line(s), skipped: " + ", ".join(kws))
        n_sol = sum(1 for w in warnings if w.startswith("SOLENOID_AS_DRIFT"))
        if n_sol:
            subs = subs + ["SOLENOID"] * 0             # counted through the warning; kept for the audit
        if subs:
            warnings.append(f"LightWin has no Envelope3D model for {len(subs)} element(s), propagated as drifts: "
                            + ", ".join(subs[:8]) + (" …" if len(subs) > 8 else ""))
        meta = {"workdir": str(wd), "lightwin_version": data.get("lightwin_version"), "python": data.get("python"),
                "tool": data.get("tool"), "phase_policy": data.get("phase_policy"),
                "n_steps_per_cell": data.get("n_steps_per_cell"), "kinds": list(data.get("kinds", [])),
                "models": [row.get("model") for row in data.get("elements", [])],
                "substituted": subs, "dropped": dropped, "ignored_commands": ignored,
                "solenoids_as_drifts": int(next((int(w.split(":")[1].split()[0]) for w in warnings
                                                if w.startswith("SOLENOID_AS_DRIFT")), 0)),
                "negative_charge_emulated": any(w.startswith("NEGATIVE_CHARGE_EMULATED") for w in warnings),
                "params": list(data.get("params", [])),
                "n_mesh": data.get("n_mesh"),
                "p0_model": "follows the field maps"}
        return OracleResult(engine="lightwin", basis=Basis.TRACEWIN, names=names,
                            length=np.asarray(data["length"], dtype=float),
                            s_out=np.asarray(data["s_out"], dtype=float), R_elem=R,
                            ref_kinetic_eV_in=np.asarray(data["w_in_ev"], dtype=float),
                            ref_kinetic_eV_out=np.asarray(data["w_out_ev"], dtype=float),
                            mass_eV=float(mass), charge=int(charge), warnings=warnings, meta=meta)
