"""FLAME oracle (PLAN §6 task 3.4).

FLAME's Python bindings (`flame-code <https://pypi.org/project/flame-code/>`_) are a
compiled extension.  Like :mod:`lattix.oracles.pytao` this adapter never insists on
importing them in-process: it resolves an interpreter that can ``import flame``, in
this order (cached on the class):

1. ``import flame`` succeeds here → run in-process;
2. ``LATTIX_FLAME_PYTHON`` — explicit interpreter path;
3. ``LATTIX_FLAME_ENV`` (default ``lattix``) looked up under the usual conda roots;
4. ``conda run -n <env> python -c "import flame, sys; print(sys.executable)"``.

The worker is a self-contained script (:data:`WORKER_SOURCE`) written into the work
directory and run with ``-I``, so it needs nothing from lattix and this module's own
name cannot shadow the real ``flame`` package.

What FLAME reports
------------------
``Machine.propagate(state, 0, N, observe=range(N))`` yields ``(index, state)`` after
every element; the state carries

* ``transmat`` — the 7×7 map of that element **per charge state** (the 6×6 block is
  the linear map, the 7th column the constant/dipole kick, ``src/moment.cpp:342``);
* ``ref_IonEk`` [eV/u] and ``ref_SampleFreq`` [Hz] (``src/moment.cpp:425,443``);
* ``pos`` [m] — the exit position (``src/moment.cpp:698`` adds ``L`` in metres).

Basis
-----
:data:`~lattix.oracles.base.Basis.FLAME` = (x [mm], x′ [rad], y [mm], y′ [rad],
φ [rad], ΔE_k [MeV/u]).  φ is measured against ``SampleFreq`` (default 80.5 MHz,
``src/flame/moment.h:17``) — **not** against any cavity frequency — and is
*late-positive*: every element sets

.. code-block:: none

    transfer(PS_S, PS_PS) = −2π·L[mm] / (SampleLambda[mm] · IonEs[MeV/u] · (βγ)³)

(``src/moment.cpp:1119`` for the quadrupole, ``:1329`` for the solenoid, …), which
after the basis transform is exactly ``R56_common = +L/γ²`` when
``basis._Z_SIGN[Basis.FLAME] = −1``.  ``tests/oracles/test_flame_adapter.py`` measures
the same number with the engine.

Because FLAME is a **per-nucleon** code, this adapter reports ``mass_eV`` and
``ref_kinetic_eV_*`` *per nucleon* (``IonEs``/``IonEk``).  β and γ are identical
either way, and it is what :func:`lattix.oracles.basis.transform_matrix` needs for the
ΔE_k [MeV/u] → δ row: ``OracleResult.to_common()`` does not pass its
``mass_per_nucleon_eV`` argument, so reporting per-nucleon values is the only way the
existing transform is right.  The total mass, the mass number and the charge state are
in ``meta``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np

from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register

_CHECK = "import flame, sys; print(sys.executable)"

#: Self-contained worker.  Written next to the deck and run with ``python -I`` so it
#: imports only ``flame``, ``numpy`` and the standard library.
WORKER_SOURCE = r'''
"""lattix FLAME worker: propagate a GLPS deck and dump per-element optics as JSON."""
import json
import sys

import numpy as np


def run(deck_path, extra_globals=None):
    import flame
    from flame import Machine

    text = open(deck_path, "rb").read()
    if extra_globals:
        text = text + b"\n" + extra_globals.encode("utf-8")
    # GLPS resolves dir()/file() against parse_context::cwd, which must be a
    # DIRECTORY (src/glps_ops.cpp: canonical(inp, ctxt->cwd)) -- not the deck file.
    import os
    M = Machine(text, path=os.path.dirname(os.path.abspath(deck_path)) or ".")
    n = len(M)
    conf = M.conf()
    elements = list(conf.get("elements", []))

    state = M.allocState({})
    # allocState() leaves ref_IonEk at 0 until the `source` element assigns the state,
    # so the machine's own IonEk is the entry energy of the first element.
    ek_start = float(conf.get("IonEk", 0.0) or np.asarray(state.ref_IonEk).ravel()[0])
    results = M.propagate(state, 0, n, observe=range(n))

    names, types, pos, ek_out, freq, mats, phis = [], [], [], [], [], [], []
    for idx, st in results:
        c = elements[idx] if idx < len(elements) else {}
        names.append(str(c.get("name", "elem%d" % idx)))
        types.append(str(c.get("type", "")))
        pos.append(float(st.pos))
        ek_out.append(float(np.asarray(st.ref_IonEk).ravel()[0]))
        freq.append(float(np.asarray(st.ref_SampleFreq).ravel()[0]))
        phis.append(float(np.asarray(st.ref_phis).ravel()[0]))
        # transmat is (7, 7, n_charge_states) -- the charge-state index is LAST
        # (measured 2026-09-03 with flame 1.9.2); the reference state is the first.
        tm = np.asarray(st.transmat, dtype=float)
        if tm.ndim == 3:
            tm = tm[:, :, 0]
        mats.append(tm.reshape(7, 7).tolist())

    ek_in = [ek_start] + ek_out[:-1]
    lengths, prev = [], 0.0
    for p in pos:
        lengths.append(p - prev)
        prev = p

    states = list(np.asarray(conf.get("IonChargeStates", [1.0]), dtype=float).ravel())
    return {
        "names": names, "types": types, "s_out": pos, "length": lengths,
        "ref_ionek_in": ek_in, "ref_ionek_out": ek_out, "sample_freq": freq,
        "ref_phis": phis, "transmat7": mats,
        "ion_es": float(conf.get("IonEs", 931.49432e6)),
        "ion_ek": float(conf.get("IonEk", ek_start)),
        "charge_states": states,
        "sim_type": str(conf.get("sim_type", "")),
        "flame_version": getattr(flame, "__version__", "?"),
        "numpy_version": np.__version__,
        "n_elements": n,
    }


def main(argv):
    args = json.loads(open(argv[1]).read())
    out = run(args["deck"], args.get("extra_globals"))
    with open(args["out"], "w") as fp:
        json.dump(out, fp)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
'''


def _can_import_flame(python: str, timeout: float = 120.0) -> tuple[bool, str]:
    try:
        proc = subprocess.run([python, "-I", "-c", _CHECK], capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"{python}: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        return False, f"{python}: {tail[-1] if tail else 'exit ' + str(proc.returncode)}"
    return True, proc.stdout.strip().splitlines()[-1]


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
    """Interpreter that imports flame, plus why every candidate was rejected."""
    tried: list[str] = []
    explicit = os.environ.get("LATTIX_FLAME_PYTHON")
    if explicit:
        ok, why = _can_import_flame(explicit)
        if ok:
            return why, tried
        tried.append(f"LATTIX_FLAME_PYTHON {why}")
    exe = "python.exe" if sys.platform == "win32" else "bin/python"
    for root in _conda_roots():
        cand = root / "envs" / env / exe
        if cand.exists():
            ok, why = _can_import_flame(str(cand))
            if ok:
                return why, tried
            tried.append(why)
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        try:
            proc = subprocess.run([conda, "run", "-n", env, "python", "-c", _CHECK],
                                  capture_output=True, text=True, timeout=300, check=False)
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
class FlameOracle:
    """FLAME (FRIB Linear Accelerator Modelling Engine) as a linear-optics oracle."""

    name = "flame"
    formats = ("flame",)

    #: (ok, reason, python path or None, in-process?) — resolved once per process
    _resolved: ClassVar[tuple[bool, str, str | None, bool] | None] = None

    @classmethod
    def reset_cache(cls) -> None:
        cls._resolved = None

    @classmethod
    def _resolve(cls) -> tuple[bool, str, str | None, bool]:
        if cls._resolved is None:
            try:
                import flame  # noqa: F401
                cls._resolved = (True, "flame in-process", sys.executable, True)
            except Exception:  # noqa: BLE001
                env = os.environ.get("LATTIX_FLAME_ENV", "lattix")
                python, tried = _resolve_python(env)
                if python:
                    cls._resolved = (True, f"flame via {python}", python, False)
                else:
                    why = ("no interpreter with flame found (pip install flame-code; note it "
                           "ships manylinux x86_64 wheels only — on macOS/arm64 it must be built "
                           "from source.  Set LATTIX_FLAME_PYTHON or LATTIX_FLAME_ENV; tried: "
                           f"{'; '.join(tried) or 'nothing'})")
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
            raise RuntimeError(f"flame oracle unavailable: {why}")
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_flame_"))
        wd.mkdir(parents=True, exist_ok=True)

        extra = _beam_globals(beam)
        if extra:
            # FLAME rejects a second definition of a global ("Name 'IonEs' already defined"):
            # a deck that already carries its beam keeps it; only the missing ones are appended
            defined = set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", deck.read_text(errors="replace"), re.M))
            kept = [ln for ln in extra.splitlines() if ln.split("=", 1)[0].strip() not in defined]
            extra = "\n".join(kept) + "\n" if kept else None
        if inprocess:
            data = _run_inprocess(str(deck), extra)
        else:
            data = self._run_subprocess(python, deck, wd, extra)
        return self._to_result(data, wd, python, inprocess, extra)

    @staticmethod
    def _run_subprocess(python: str, deck: Path, wd: Path, extra: str | None) -> dict:
        script = wd / "lattix_flame_worker.py"
        script.write_text(WORKER_SOURCE)
        out = wd / "flame_result.json"
        args = wd / "flame_args.json"
        args.write_text(json.dumps({"deck": str(deck), "out": str(out), "extra_globals": extra}))
        cmd = [python, "-I", str(script), str(args)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(deck.parent),
                              check=False)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"flame worker failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                               f"--- stderr ---\n{proc.stderr[-4000:]}\n--- stdout ---\n"
                               f"{proc.stdout[-2000:]}")
        return json.loads(out.read_text())

    # ------------------------------------------------------------------
    @staticmethod
    def _to_result(data: dict, wd: Path, python: str | None, inprocess: bool,
                   extra: str | None) -> OracleResult:
        names = [str(x) for x in data["names"]]
        n = len(names)
        R7 = np.asarray(data["transmat7"], dtype=float).reshape(n, 7, 7)
        R = R7[:, :6, :6].copy()
        mass_per_u = float(data["ion_es"])
        states = [float(x) for x in data.get("charge_states") or [1.0]]
        q_over_a = states[0] if states else 1.0
        charge, mass_number = _charge_and_mass_number(q_over_a)
        warnings: list[str] = []
        if len(states) > 1:
            warnings.append(f"deck has {len(states)} charge states; the reference is the first "
                            f"(Q/A = {q_over_a:.9g}) and transmat[0] is reported")
        meta = {
            "python": python, "mode": "in-process" if inprocess else "subprocess",
            "flame_version": data.get("flame_version"),
            "numpy_version": data.get("numpy_version"), "sim_type": data.get("sim_type"),
            "workdir": str(wd), "types": [str(t) for t in data.get("types", [])],
            "charge_states": states, "mass_number": mass_number,
            "mass_total_eV": mass_per_u * mass_number,
            "ref_phis_rad": [float(x) for x in data.get("ref_phis", [])],
            "dipole_column": R7[:, :6, 6].tolist(),
            "units": ("mass_eV and ref_kinetic_eV_* are PER NUCLEON (FLAME's IonEs/IonEk in "
                      "eV/u); multiply by meta['mass_number'] for the total"),
            "p0_model": ("follows p0: rfcavity and stripper change ref_IonEk; the 6th "
                         "coordinate is ΔE_k [MeV/u], normalised by the local β²γ"),
            "phase_reference": ("φ [rad] is measured against SampleFreq, not a cavity "
                                "frequency (src/flame/moment.h:17,52)"),
            "beam_override": extra,
        }
        return OracleResult(
            engine="flame", basis=Basis.FLAME, names=names,
            length=np.asarray(data["length"], dtype=float),
            s_out=np.asarray(data["s_out"], dtype=float),
            R_elem=R,
            ref_kinetic_eV_in=np.asarray(data["ref_ionek_in"], dtype=float),
            ref_kinetic_eV_out=np.asarray(data["ref_ionek_out"], dtype=float),
            mass_eV=mass_per_u, charge=charge,
            rf_frequency_Hz=np.asarray(data["sample_freq"], dtype=float),
            warnings=warnings, meta=meta,
        )


def cavity_data_dir() -> Path | None:
    """FLAME's shipped RF-cavity tables (``axisData_*.txt``, ``Multipole*/``).

    Decks written by FRIB say ``Eng_Data_Dir = dir("data")`` and expect the
    directory next to the deck; the installed package keeps a copy under
    ``flame/test/data``, which is what the tests point ``Eng_Data_Dir`` at.
    """
    ok, _, python, inprocess = FlameOracle._resolve()
    if not ok:
        return None
    if inprocess:
        import flame
        base = Path(flame.__file__).parent
    else:
        proc = subprocess.run([python, "-I", "-c",
                               "import flame, os; print(os.path.dirname(flame.__file__))"],
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        base = Path(proc.stdout.strip().splitlines()[-1])
    for cand in (base / "test" / "data", base / "data"):
        if (cand / "axisData_41.txt").is_file():
            return cand
    return None


def _run_inprocess(deck: str, extra: str | None) -> dict:
    namespace: dict = {"__name__": "lattix_flame_worker"}
    exec(compile(WORKER_SOURCE, "<lattix flame worker>", "exec"), namespace)  # noqa: S102
    return namespace["run"](deck, extra)


def _beam_globals(beam: BeamSpec | None) -> str | None:
    """GLPS assignments appended to the deck to override its beam.

    FLAME reads ``IonEs``/``IonEk``/``IonChargeStates`` from the machine-level
    Config, which is a flat last-write-wins map, so appending them overrides the
    deck.  **FLAME is per nucleon**, so a :class:`BeamSpec` is interpreted as one
    nucleon (identical for protons, A = 1).
    """
    if beam is None:
        return None
    return (f"IonEs = {beam.mass_eV!r};\n"
            f"IonEk = {beam.kinetic_energy_eV!r};\n"
            f"IonZ = {float(beam.charge)!r};\n"
            f"IonChargeStates = [{float(beam.charge)!r}];\n"
            "NCharge = [1.0];\n")


def _charge_and_mass_number(q_over_a: float, max_a: int = 400) -> tuple[int, int]:
    from fractions import Fraction

    if q_over_a == 0.0:
        return 0, 1
    frac = Fraction(abs(q_over_a)).limit_denominator(max_a)
    q = frac.numerator or 1
    return (q if q_over_a > 0 else -q), frac.denominator


__all__ = ["FlameOracle", "WORKER_SOURCE", "cavity_data_dir"]
