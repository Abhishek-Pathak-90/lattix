"""Find a Python interpreter that can import an engine living in another conda environment.

Several engines refuse to share an environment with the rest of lattix (pytao pins python 3.13,
LightWin needs python ≥ 3.12 while env ``lattix`` is 3.11), so their adapters run a worker script in
the engine's own environment.  The resolution order, verified by actually importing the module:

1. an explicit interpreter from the environment variable ``explicit_var``;
2. the current interpreter, when it can import the module (in-process);
3. ``<conda root>/envs/<env>/bin/python`` for every conda root that can be found (``CONDA_PREFIX``,
   ``CONDA_EXE``, ``~/{anaconda3,miniconda3,miniforge3,mambaforge}``, ``/opt/...``);
4. ``conda run -n <env> python -c "import <module>, sys; print(sys.executable)"``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def can_import(python: str, module: str, timeout: float = 120.0) -> tuple[bool, str]:
    """(ok, interpreter path or the reason the import failed)."""
    check = f"import {module}, sys; print(sys.executable)"
    try:
        proc = subprocess.run([python, "-I", "-c", check], capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{python}: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"{python}: {tail[-1] if tail else 'exit ' + str(proc.returncode)}"
    return True, proc.stdout.strip().splitlines()[-1]


def conda_roots() -> list[Path]:
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


def resolve_python(module: str, env: str, explicit_var: str, *,
                   try_current: bool = True) -> tuple[str | None, list[str]]:
    """Interpreter that imports *module*, plus the reasons every candidate was rejected."""
    tried: list[str] = []
    explicit = os.environ.get(explicit_var)
    if explicit:
        ok, why = can_import(explicit, module)
        if ok:
            return why, tried
        tried.append(f"{explicit_var} {why}")
    if try_current:
        ok, why = can_import(sys.executable, module)
        if ok:
            return why, tried
        tried.append(why)
    exe = "python.exe" if sys.platform == "win32" else "bin/python"
    for root in conda_roots():
        cand = root / "envs" / env / exe
        if cand.exists():
            ok, why = can_import(str(cand), module)
            if ok:
                return why, tried
            tried.append(why)
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        check = f"import {module}, sys; print(sys.executable)"
        try:
            proc = subprocess.run([conda, "run", "-n", env, "python", "-c", check],
                                  capture_output=True, text=True, timeout=300, check=False)
            lines = proc.stdout.strip().splitlines()
            if proc.returncode == 0 and lines and Path(lines[-1]).exists():
                return lines[-1], tried
            err = (proc.stderr or proc.stdout).strip().splitlines()
            last = err[-1] if err else f"exit {proc.returncode}"
            tried.append(f"conda run -n {env}: {last}")
        except (OSError, subprocess.TimeoutExpired) as e:
            tried.append(f"conda run -n {env}: {e}")
    else:
        tried.append("no conda executable on PATH / CONDA_EXE")
    return None, tried
