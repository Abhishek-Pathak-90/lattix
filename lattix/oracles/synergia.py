"""Synergia oracle: per-element first-order maps of a Synergia lattice JSON, evaluated by Synergia 3's own
propagator in its environment through :mod:`lattix.oracles.synergia_worker`.

The interpreter is found through ``LATTIX_SYNERGIA_PYTHON``, the current interpreter, or the clone's pixi
install (``LATTIX_SYNERGIA_ROOT`` or the local clone: ``.pixi/envs/cpu/bin/python`` with ``install_pixi``'s
site-packages on ``PYTHONPATH``).  Conventions are measured in ``docs/oracles.md`` (Phase 5.9).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np

from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register

_WORKER = Path(__file__).with_name("synergia_worker.py")


def _pixi_python(root: Path) -> tuple[str | None, dict[str, str]]:
    """The pixi environment's interpreter and the environment that finds the install."""
    for env in ("cpu", "default"):
        py = root / ".pixi" / "envs" / env / "bin" / "python"
        if py.exists():
            install = root / "install_pixi"
            site = sorted(install.glob("lib/python3*/site-packages"))
            paths = [str(install / "lib"), *[str(x) for x in site]]
            return str(py), {"PYTHONPATH": os.pathsep.join(paths), "DYLD_LIBRARY_PATH": str(install / "lib"),
                             "LD_LIBRARY_PATH": str(install / "lib"), "OMP_PROC_BIND": "false",
                             "OMPI_MCA_shmem_mmap_enable_nfs_warning": "0"}
    return None, {}


@register
class SynergiaOracle:
    name = "synergia"
    formats = ("synergia",)

    _resolved: ClassVar[tuple[bool, str, str | None, dict] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    def available(self) -> tuple[bool, str]:
        if SynergiaOracle._resolved is None:
            SynergiaOracle._resolved = self._resolve()
        ok, why, _, _ = SynergiaOracle._resolved
        return ok, why

    def _resolve(self) -> tuple[bool, str, str | None, dict]:
        tried = []
        cands: list[tuple[str, dict]] = []
        explicit = os.environ.get("LATTIX_SYNERGIA_PYTHON")
        if explicit:
            cands.append((explicit, {}))
        cands.append((sys.executable, {}))
        root = os.environ.get("LATTIX_SYNERGIA_ROOT")          # a clone built with pixi
        if root:
            py, env = _pixi_python(Path(root).expanduser())
            if py:
                cands.append((py, env))
        for py, env in cands:
            try:
                proc = subprocess.run([py, "-P", "-c", "import synergia, sys; print(sys.executable)"],
                                      capture_output=True, text=True, timeout=120, env={**os.environ, **env},
                                      check=False)
            except (OSError, subprocess.TimeoutExpired) as e:
                tried.append(f"{py}: {e}")
                continue
            if proc.returncode == 0:
                return True, f"Synergia via {py}", py, env
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            tried.append(f"{py}: {tail[-1] if tail else 'exit ' + str(proc.returncode)}")
        return False, "no interpreter imports synergia: " + "; ".join(tried), None, {}

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None) -> OracleResult:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(f"synergia oracle unavailable: {why}")
        _, _, python, env = SynergiaOracle._resolved
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        if fmt not in (None, "synergia"):
            raise ValueError(f"Synergia reads its lattice JSON only, not {fmt!r}")
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_synergia_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "synergia_result.json"
        # -P keeps the worker's own directory off sys.path (lattix/oracles/synergia.py would shadow the package)
        cmd = [python, "-P", str(_WORKER), str(deck), "--out", str(out)]
        if beam is not None:
            cmd += ["--ke-ev", repr(float(beam.kinetic_energy_eV)), "--mass-ev", repr(float(beam.mass_eV)),
                    "--charge", repr(float(beam.charge)), "--species", str(beam.species)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wd), env={**os.environ, **env}, check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"synergia worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n{proc.stdout[-2000:]}")
        data = json.loads(out.read_text())
        names = [str(x) for x in data["names"]]
        n = len(names)
        R = np.asarray(data["R"], dtype=float).reshape(n, 6, 6)
        mass = float(beam.mass_eV) if beam is not None else float(data["mass_ev"])
        charge = int(beam.charge) if beam is not None else int(data["charge"])
        meta = {"workdir": str(wd), "synergia_version": data.get("synergia_version"), "python": data.get("python"),
                "kinds": list(data.get("kinds", [])), "p0_model": data.get("p0_model")}
        return OracleResult(engine="synergia", basis=Basis.SYNERGIA, names=names,
                            length=np.asarray(data["length"], dtype=float),
                            s_out=np.asarray(data["s_out"], dtype=float), R_elem=R,
                            ref_kinetic_eV_in=np.asarray(data["w_in_ev"], dtype=float),
                            ref_kinetic_eV_out=np.asarray(data["w_out_ev"], dtype=float),
                            mass_eV=mass, charge=charge, warnings=[str(w) for w in data.get("warnings", [])],
                            meta=meta)
