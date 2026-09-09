"""TraceWin oracle: headless batch run of the licensed TraceWin binary.

Runs ``TraceWin <project.ini> hide dat_file=… path_cal=… energy1=… freq1=…``
(the batch syntax documented in the TraceWin manual and used by LightWin),
then reads two of its outputs:

* ``Transfer_matrix1.dat`` — cumulative 6×6 maps ``ELE# n : s m`` + 6 rows,
  in (x, x', y, y', z, dp/p) with lengths in metres (a 50 mm drift shows
  R12 = 0.05 and R56 = +L/γ², i.e. z is ahead-positive); per-element maps are
  recovered as ``M_n · M_{n-1}⁻¹`` (HELIX ``test_tracewin_crosscheck.py:302``).
* ``tracewin.out`` — one row per element exit with ``gama-1`` → reference
  kinetic energy (TraceWin follows p0 through cavities).

Local-only oracle: needs ``TRACEWIN_EXE`` (or the default app path) and a
licence.  The trial binary occasionally stalls instead of exiting (seen once in a
600 s full-suite run; the same deck passes in 16 s alone), so the default ``timeout`` is
120 s — small decks finish in seconds.  The trial build on this Mac stops at 20 elements — the adapter
raises a clear error when TraceWin reports that limit.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from lattix.oracles.base import Basis, BeamSpec, OracleResult, Probe, register

_PROJECT_INI = Path(__file__).with_name("data") / "tracewin_project.ini"
_ELE_RE = re.compile(r"^\s*ELE#\s*(\d+)\s*:\s*([-+0-9.eE]+)\s*m")


def tracewin_exe() -> Path | None:
    """The TraceWin binary named by ``TRACEWIN_EXE``, or None when it is unset or not executable."""
    v = os.environ.get("TRACEWIN_EXE")
    if not v:
        return None
    p = Path(v).expanduser()
    return p if p.is_file() and os.access(p, os.X_OK) else None


def read_transfer_matrix_file(path: Path) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Parse ``Transfer_matrix1.dat`` → (element numbers, s [m], cumulative maps (N,6,6))."""
    nums, s, mats = [], [], []
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        m = _ELE_RE.match(lines[i])
        if m:
            nums.append(int(m.group(1)))
            s.append(float(m.group(2)))
            rows = [list(map(float, lines[i + k].split())) for k in range(1, 7)]
            mats.append(np.array(rows, dtype=float))
            i += 7
        else:
            i += 1
    if not mats:
        raise ValueError(f"no 'ELE#' blocks found in {path}")
    return nums, np.array(s), np.array(mats)


def read_tracewin_out(path: Path) -> dict[str, np.ndarray]:
    """Parse ``tracewin.out`` (the ``##`` header names the columns; one row per element exit,
    row 0 = lattice entrance)."""
    header, rows = None, []
    for line in Path(path).read_text().splitlines():
        if line.lstrip().startswith("##"):
            header = line.lstrip("# ").split()
            continue
        if header and line.strip() and line.split()[0].isdigit():
            rows.append([float(x) for x in line.split()[1:len(header) + 1]])
    if header is None:
        raise ValueError(f"no '##' column header in {path}")
    arr = np.array(rows, dtype=float)
    return {name: arr[:, j] for j, name in enumerate(header) if j < arr.shape[1]}


@register
class TracewinOracle:
    name = "tracewin"
    formats = ("tracewin",)

    def available(self) -> tuple[bool, str]:
        exe = tracewin_exe()
        if exe is None:
            return False, "TRACEWIN_EXE is not set, or is not an executable file"
        if not _PROJECT_INI.is_file():
            return False, f"bundled project file missing: {_PROJECT_INI}"
        return True, f"{exe} (licence permitting; trial builds stop at 20 elements)"

    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            nbr_part: int = 1000, timeout: float = 120.0) -> OracleResult:
        exe = tracewin_exe()
        if exe is None:
            raise RuntimeError("TraceWin binary not available")
        beam = beam or BeamSpec()
        wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="lattix_tw_"))
        wd.mkdir(parents=True, exist_ok=True)
        out = wd / "out"
        out.mkdir(exist_ok=True)
        local_deck = wd / Path(deck).name
        shutil.copy(deck, local_deck)
        local_ini = wd / "project.ini"                # TraceWin rewrites its project file: never the repo copy
        shutil.copy(_PROJECT_INI, local_ini)
        freq_mhz = (beam.frequency_Hz or 352.21e6) / 1e6
        args = [str(exe), str(local_ini), "hide", f"dat_file={local_deck}", f"path_cal={out}",
                f"energy1={beam.kinetic_energy_eV * 1e-6:.12g}", "current1=0",
                f"freq1={freq_mhz:.12g}", f"nbr_part1={nbr_part}"]
        if beam.species.lower() != "proton":
            args += [f"mass1={beam.mass_eV * 1e-6:.12g}", f"charge1={beam.charge}"]
        proc = None
        for attempt in range(2):          # the trial build stalls now and then; one retry
            try:
                proc = subprocess.run(args, cwd=wd, capture_output=True, text=True, timeout=timeout)
                break
            except subprocess.TimeoutExpired:
                if attempt == 1:
                    raise
        log = (proc.stdout or "") + (proc.stderr or "")
        if "Limited trial version" in log:
            raise RuntimeError(f"TraceWin trial licence limit hit: {log.strip().splitlines()[:2]}")
        tm = out / "Transfer_matrix1.dat"
        if not tm.is_file():
            raise RuntimeError("TraceWin produced no Transfer_matrix1.dat "
                               f"(rc={proc.returncode}):\n{log[-2000:]}")
        nums, s_out, cum = read_transfer_matrix_file(tm)
        n = len(nums)
        R = np.empty_like(cum)
        prev = np.eye(6)
        for i in range(n):
            R[i] = cum[i] @ np.linalg.inv(prev)
            prev = cum[i]
        lengths = np.diff(np.concatenate([[0.0], s_out]))
        cols = read_tracewin_out(out / "tracewin.out")
        gm1 = cols["gama-1"]
        ke_exit = gm1 * beam.mass_eV            # row 0 = entrance, rows 1..N = exits
        if len(ke_exit) < n + 1:
            raise RuntimeError(f"tracewin.out has {len(ke_exit)} rows for {n} matrix blocks")
        w_in = ke_exit[:n]
        w_out = ke_exit[1:n + 1]
        names = self._element_tags(local_deck, n)
        warnings = [ln for ln in log.splitlines() if "warn" in ln.lower() or "error" in ln.lower()]
        return OracleResult(
            engine=self.name, basis=Basis.TRACEWIN, names=names, length=lengths, s_out=s_out,
            R_elem=R, ref_kinetic_eV_in=w_in, ref_kinetic_eV_out=w_out,
            mass_eV=beam.mass_eV, charge=beam.charge,
            rf_frequency_Hz=np.full(n, freq_mhz * 1e6), warnings=warnings,
            meta={"exe": str(exe), "workdir": str(wd), "args": args[2:], "probe": "not implemented"},
        )

    @staticmethod
    def _element_tags(deck: Path, n: int) -> list[str]:
        """Best-effort names: ``<CARD>_<k>`` from the deck's element cards in order."""
        skip = {"FREQ", "END", "TITLE", "PARTRAN_STEP", "LATTICE", "LATTICE_END", "FIELD_MAP_PATH",
                "SET_SYNC_PHASE", "ERROR_", "ADJUST", "SET_", "MIN_", "SUPERPOSE_MAP", "SPACE_CHARGE_COMP"}
        tags = []
        for raw in deck.read_text(encoding="latin-1").splitlines():
            line = raw.split(";")[0].strip()
            if not line:
                continue
            tok = line.split()
            kw = tok[0].upper()
            if ":" in kw and len(tok) > 1:
                kw = tok[1].upper()
            if any(kw == s or kw.startswith(s) for s in skip):
                continue
            tags.append(kw)
        if len(tags) != n:
            tags = [f"ELE{i + 1}" for i in range(n)]
        else:
            tags = [f"{t}_{i + 1}" for i, t in enumerate(tags)]
        return tags
