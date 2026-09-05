"""Bmad/Tao oracle, driven through :mod:`lattix.oracles.bmad_worker`.

pytao (and Bmad's shared library) lives in a separate conda environment
(``bmad``: Python 3.13, numpy 2) that cannot be merged with the base one, so
this adapter never imports pytao itself unless it happens to be importable
in-process.  Otherwise it resolves an interpreter that can import pytao, in
this order (the answer is cached on the class):

1. ``import pytao`` succeeds here → run the worker in-process;
2. ``LATTIX_BMAD_PYTHON`` — explicit interpreter path;
3. ``LATTIX_BMAD_ENV`` (default ``bmad``) looked up as
   ``$CONDA_PREFIX/envs/<env>``, ``$CONDA_PREFIX/../<env>``, the root of
   ``$CONDA_EXE``, ``~/{anaconda3,miniconda3,miniforge3,mambaforge}/envs/<env>``,
   ``/opt/{conda,anaconda3,miniconda3}/envs/<env>``;
4. ``conda run -n <env> python -c "import pytao, sys; print(sys.executable)"``.

Every candidate is verified by actually importing pytao in it.  The worker is
started as ``<python> -I <worker file path>`` (isolated mode: neither the
caller's ``PYTHONPATH`` nor the worker's own directory — which holds this
``pytao.py`` — can shadow the real ``pytao`` package).

Result: :class:`OracleResult` in ``Basis.BMAD`` = (x, px, y, py, z, pz), z
ahead-positive, pz = Δp/p0 with the *local* p0 (Bmad's ``lcavity`` follows p0,
so ``ref_kinetic_eV_out`` differs from ``ref_kinetic_eV_in`` across cavities).
``mass_eV``/``charge`` are Bmad's own (``mass_of``/``charge_of``), which differ
from :data:`lattix.oracles.base.SPECIES` at the 1e-9 level (m_p 938 272 089.43
vs 938 272 088.16 eV) — see ``meta["mass_source"]``.
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
from lattix.oracles.envs import can_import, conda_roots, resolve_python

_WORKER = Path(__file__).with_name("bmad_worker.py")
_CHECK = "import pytao, sys; print(sys.executable)"


def _can_import_pytao(python: str, timeout: float = 120.0) -> tuple[bool, str]:
    return can_import(python, "pytao", timeout)


def _conda_roots() -> list[Path]:
    return conda_roots()


def _resolve_python(env: str) -> tuple[str | None, list[str]]:
    """Interpreter that imports pytao, plus the reasons every candidate was rejected."""
    return resolve_python("pytao", env, "LATTIX_BMAD_PYTHON", try_current=False)


@register
class BmadOracle:
    name = "bmad"
    formats = ("bmad",)

    #: (ok, reason, python path or None, in-process?) — resolved once per process
    _resolved: ClassVar[tuple[bool, str, str | None, bool] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    @classmethod
    def _resolve(cls) -> tuple[bool, str, str | None, bool]:
        if cls._resolved is None:
            try:
                import pytao  # noqa: F401
                cls._resolved = (True, "pytao in-process", sys.executable, True)
            except Exception:  # noqa: BLE001
                env = os.environ.get("LATTIX_BMAD_ENV", "bmad")
                python, tried = _resolve_python(env)
                if python:
                    cls._resolved = (True, f"pytao via {python}", python, False)
                else:
                    why = ("no interpreter with pytao found (set LATTIX_BMAD_PYTHON or "
                           f"LATTIX_BMAD_ENV; tried: {'; '.join(tried) or 'nothing'})")
                    cls._resolved = (False, why, None, False)
        return cls._resolved

    def available(self) -> tuple[bool, str]:
        ok, why, _, _ = self._resolve()
        return ok, why

    # ------------------------------------------------------------------
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None) -> OracleResult:
        ok, why, python, inprocess = self._resolve()
        if not ok:
            raise RuntimeError(f"bmad oracle unavailable: {why}")
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_bmad_"))
        wd.mkdir(parents=True, exist_ok=True)
        opt = beam or BeamSpec()  # optics defaults; energy lines only for an explicit beam

        probe_npy = None
        if probe is not None:
            from lattix.oracles.basis import transform_matrix

            # common -> Bmad is the identity by construction; evaluated with the
            # requested beam (the deck's energy is only known after the run)
            Tinv = np.linalg.inv(transform_matrix(Basis.BMAD, opt.kinetic_energy_eV, opt.mass_eV))
            probe_npy = wd / "probe_bmad.npy"
            np.save(probe_npy, np.asarray(probe.coords, dtype=float) @ Tinv.T)

        kwargs = dict(species=(beam.species if beam else None),
                      kinetic_ev=(float(beam.kinetic_energy_eV) if beam else None),
                      betx=opt.betx, alfx=opt.alfx, bety=opt.bety, alfy=opt.alfy,
                      dx=opt.dx, dpx=opt.dpx, dy=opt.dy, dpy=opt.dpy,
                      probe=(str(probe_npy) if probe_npy else None), workdir=str(wd))
        if inprocess:
            from lattix.oracles import bmad_worker

            data = bmad_worker.run(str(deck), **kwargs)
        else:
            data = self._run_subprocess(python, deck, wd, kwargs)
        return self._to_result(data, wd, python, inprocess)

    @staticmethod
    def _run_subprocess(python: str, deck: Path, wd: Path, kw: dict) -> dict:
        out = wd / "bmad_result.json"
        cmd = [python, "-I", str(_WORKER), str(deck), "--out", str(out), "--workdir", str(wd)]
        for k in ("betx", "alfx", "bety", "alfy", "dx", "dpx", "dy", "dpy"):
            cmd += [f"--{k}", repr(float(kw[k]))]
        if kw["kinetic_ev"] is not None:
            cmd += ["--species", kw["species"] or "proton", "--kinetic-ev", repr(kw["kinetic_ev"])]
        if kw["probe"]:
            cmd += ["--probe", kw["probe"]]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(wd), check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"bmad worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n"
                               f"{proc.stdout[-2000:]}")
        return json.loads(out.read_text())

    @staticmethod
    def _to_result(data: dict, wd: Path, python: str | None, inprocess: bool) -> OracleResult:
        names = [str(x) for x in data["names"]]
        n = len(names)
        mass = float(data["mass_ev"])
        charge = int(round(float(data["charge"])))
        e_in = np.asarray(data["e_tot_in"], dtype=float)
        e_out = np.asarray(data["e_tot_out"], dtype=float)
        R = np.asarray(data["mat6"], dtype=float).reshape(n, 6, 6)
        col = {k: np.asarray(v, dtype=float) for k, v in data["twiss"].items()}
        twiss = {k: col[k] for k in ("betx", "alfx", "bety", "alfy")}
        disp = {k: col[k] for k in ("dx", "dpx", "dy", "dpy")}
        survey = np.asarray(data["floor"], dtype=float).reshape(n, 4)
        probe_out = data.get("probe_out")
        if probe_out is not None:
            probe_out = np.asarray(probe_out, dtype=float)
        meta = {
            "python": python, "mode": "in-process" if inprocess else "subprocess",
            "tao_version": data.get("tao_version"), "pytao_version": data.get("pytao_version"),
            "numpy_version": data.get("numpy_version"), "wrapper": data.get("wrapper"),
            "species": data.get("species"), "element_keys": list(data.get("keys", [])),
            "mass_source": "Bmad mass_of()/charge_of() (differs from lattix SPECIES at ~1e-9)",
            "p0_model": "follows p0: lcavity changes the reference momentum; pz is relative "
                        "to the local p0",
            "p0c_in": data.get("p0c_in"), "p0c_out": data.get("p0c_out"), "workdir": str(wd),
        }
        return OracleResult(
            engine="bmad", basis=Basis.BMAD, names=names,
            length=np.asarray(data["length"], dtype=float),
            s_out=np.asarray(data["s_out"], dtype=float),
            R_elem=R, ref_kinetic_eV_in=e_in - mass, ref_kinetic_eV_out=e_out - mass,
            mass_eV=mass, charge=charge, twiss=twiss, disp=disp, survey=survey,
            probe_out=probe_out, warnings=[str(w) for w in data.get("warnings", [])], meta=meta,
        )
