"""Elegant oracle: a generated ``.ele`` command file around the user's ``.lte``
deck; SDDS outputs parsed with ``pysdds`` or with the SDDS command-line tools.

What elegant reports (measured 2026-09-03, elegant 2026.3.0 from conda-forge):

* ``matrix_output individual_matrices=1`` gives one row per beamline element
  in order (plus a synthetic ``_BEG_`` MARK row, dropped here) holding that
  element's own first-order map ``R11..R66`` (entry -> exit, on the reference
  trajectory), the exit position ``s`` and the exit reference momentum
  ``pCentral`` (βγ).  ``individual_matrices=0`` gives start -> exit cumulative
  maps instead (``matrix_mode="cumulative"``).
* Native basis (x, x', y, y', s, δ) with δ = Δp/p0 and **s = path length**
  (elegant defines s = βct with the particle's own velocity and recomputes s
  whenever the velocity changes).  Consequently a drift has R56 = 0 at any
  energy (no velocity term L/γ²), and the RFCA matrix computes the phase
  slip as ω·Δs/c rather than ω·Δs/(βc) — both exact only for β -> 1.  The
  adapter returns the matrices exactly as elegant computes them (PLAN §5.1
  forbids silent convention fixes); the numbers are pinned in
  ``tests/oracles/test_elegant_adapter.py``.
* ``twiss_output matched=0`` propagates the BeamSpec Twiss and dispersion;
  its rows (and those of ``floor_coordinates``: X, Y, Z, theta at element
  exits) are 1:1 with the matrix rows.
* Reference momentum: ``run_setup p_central`` (βγ) is set from the BeamSpec.
  ``p_central_mev`` is the same quantity in MeV/c (verified: it is a
  momentum, not an energy) but goes through elegant's internal mass
  constants, so βγ is written instead.  Every species goes through
  ``change_particle name="custom", mass_ratio = m / m_e(elegant)`` with
  m_e(elegant) = 0.51099906 MeV (CODATA-86, measured from ``pCentral``), so
  the mass elegant uses equals ``SPECIES``; the built-ins would not: the
  default electron is 2.2e-7 above CODATA-2018 and ``name="proton"`` is
  1836.181 m_e = 938.2866 MeV (1.5e-5 off).  ``charge_ratio`` is relative to
  the *electron's* charge: electron +1, proton -1, H⁻ +1.  **The RFCA energy
  gain follows the charge sign relative to the electron**: negative particles
  gain +V·sin(PHASE) (crest at PHASE = 90°), positive particles gain
  -V·sin(PHASE) (crest at PHASE = -90°).
* ``RFCA change_p0=1`` updates ``pCentral`` after the cavity (p0-following);
  ``ref_kinetic_eV_out`` is derived from that column, ``ref_kinetic_eV_in``
  from the previous exit (first element: the ``_BEG_`` row).
* Probe: elegant tracks (x, x', y, y', t [s], p = βγ), not (s, δ).  Input
  conversion ``t = s_native / (β0 c)``, ``p = βγ0 (1 + δ)`` (elegant's own
  s = βct relation).  Output ``s_native = β_ref c (t - t_ref)`` (late-positive,
  arrival-time based) and ``δ = p / p_ref - 1``, where the reference particle
  is tracked *alone* as the fiducial first step (``run_control
  first_is_fiducial=1, reset_rf_for_each_step=0``) so RF phases and the
  momentum profile come from it, not from the probe centroid.  Note that this
  arrival-time s carries the velocity term that elegant's matrices lack.
* The conda-forge package ships no ``defns.rpn``; elegant refuses to start
  without one, so a copy of the standard constant/function definitions is
  written next to the command file unless ``RPN_DEFNS`` points at a file.

Binary lookup: ``$ELEGANT_EXE``, then ``$LATTIX_ENV_BIN/elegant``, then
``~/anaconda3/envs/lattix/bin/elegant``, then ``elegant`` on PATH.  SDDS
files are read with ``pysdds`` when importable, otherwise with
``sdds2stream`` from the elegant binary's directory (``sdds_backend`` selects
one explicitly).
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from lattix.oracles.base import C_LIGHT, Basis, BeamSpec, OracleResult, Probe, register

#: elegant's own electron rest mass (``me_mev`` = 0.51099906, CODATA-86), measured
#: as ``p_central_mev / pCentral`` for the default electron.  ``mass_ratio`` in
#: ``change_particle`` multiplies this value.
ME_ELEGANT_EV = 510_999.06

_DEFAULT_ENV_BIN = Path("~/anaconda3/envs/lattix/bin").expanduser()
_BEGIN_MARK = "_BEG_"
_R_COLS = [f"R{i}{j}" for i in range(1, 7) for j in range(1, 7)]
_TWI_COLS = {"betx": "betax", "alfx": "alphax", "bety": "betay", "alfy": "alphay",
             "dx": "etax", "dpx": "etaxp", "dy": "etay", "dpy": "etayp"}
_FLR_COLS = ["X", "Y", "Z", "theta"]

# Constants and helper functions from the SDDS toolkit's standard ``defns.rpn``
# (the statistics / test functions are omitted; values are the originals so
# expressions in existing decks evaluate identically).
_DEFNS_RPN = """\
/* rpn definitions written by lattix (subset of the SDDS toolkit defns.rpn)
1 atan 4 * sto pi pop
10 ln sto log_10 pop
100 exp sto HUGE pop
HUGE sto on_div_by_zero pop
udf
chs
-1 *

udf
abs
0 < pop ? chs : 0 + $

udf
mod
= rup swap = rup swap = rup / int rdn * rdn swap - 0 < ? pop rdn + : pop rdn pop $

udf
tan
= cos swap sin swap /

udf
dsin
180 / pi * sin

udf
dcos
180 / pi * cos

udf
dtan
180 / pi * tan

udf
dasin
asin 180 * pi /

udf
dacos
acos 180 * pi /

udf
datan
atan 180 * pi /

udf
rec
1 swap /

udf
rtod
180 * pi /

udf
dtor
180 / pi *

udf
cosh
exp = rec + 2 /

udf
sinh
exp = chs rec + 2 /

udf
tanh
= sinh swap cosh /

udf
acosh
= sqr 1 - sqrt + ln

udf
asinh
= sqr 1 + sqrt + ln

udf
atanh
= 1 + swap chs 1 + / sqrt ln

udf
10x
10 swap pow

udf
log
ln log_10 /

udf
hypot
sqr swap sqr + sqrt

udf
max2
< ? swap pop : pop $

udf
min2
> ? swap pop : pop $

udf
true
1 1 ==

udf
false
1 0 ==

2.99792458e10 sto c_cgs pop
2.99792458e8  sto c_mks pop
4.80325e-10 sto e_cgs pop
1.60217733e-19 sto e_mks pop
9.1093897e-28 sto me_cgs pop
9.1093897e-31 sto me_mks pop
2.81794092e-13 sto re_cgs pop
2.81794092e-15 sto re_mks pop
1.380658e-16 sto kb_cgs pop
1.380658e-23 sto kb_mks pop
0.51099906 sto mev pop
1.0545887e-34 sto hbar_mks pop
6.582173e-22 sto hbar_MeVs pop
1.6726485e-27 sto mp_mks pop
4 pi * 1e-7 * sto mu_o pop
1e7 4 / pi / c_mks sqr / sto eps_o pop
191.655e-2 sto Kas pop
75.0499e-2 sto Kaq pop
udf
beta.p
= sqr 1 + sqrt /

udf
gamma.p
sqr 1 + sqrt

udf
gamma.beta
sqr 1 swap - sqrt rec

udf
p.beta
= sqr 1 swap - sqrt /

"""


# ---------------------------------------------------------------------------
# binary discovery
# ---------------------------------------------------------------------------
def find_elegant() -> Path | None:
    """``$ELEGANT_EXE`` -> ``$LATTIX_ENV_BIN/elegant`` -> ``~/anaconda3/envs/lattix/bin/elegant``
    -> ``shutil.which("elegant")``."""
    v = os.environ.get("ELEGANT_EXE")
    if v:
        p = Path(v).expanduser()
        return p if p.is_file() else None
    cands = []
    v = os.environ.get("LATTIX_ENV_BIN")
    if v:
        cands.append(Path(v).expanduser() / "elegant")
    cands.append(_DEFAULT_ENV_BIN / "elegant")
    for p in cands:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    w = shutil.which("elegant")
    return Path(w) if w else None


@functools.lru_cache(maxsize=8)
def elegant_version(exe: str) -> str:
    """Version/date from elegant's banner (``This is elegant 2026.3.0, Jul  2 2026, by ...``)."""
    r = subprocess.run([exe], capture_output=True, text=True, timeout=60,
                       stdin=subprocess.DEVNULL)
    m = re.search(r"This is elegant ([^,]+),\s*([^,]+),", r.stdout + r.stderr)
    if not m:
        first = (r.stdout + r.stderr).strip().splitlines()
        raise RuntimeError("unrecognised elegant banner: "
                           f"{first[0] if first else '<no output>'!r}")
    return f"{m.group(1).strip()} ({' '.join(m.group(2).split())})"


def rpn_defns_path(workdir: Path) -> Path:
    """``$RPN_DEFNS`` when it names an existing file, else a generated ``defns.rpn`` in *workdir*."""
    v = os.environ.get("RPN_DEFNS")
    if v and Path(v).expanduser().is_file():
        return Path(v).expanduser().resolve()
    p = (Path(workdir) / "defns.rpn").resolve()   # absolute: elegant resolves -rpnDefns from its cwd
    p.write_text(_DEFNS_RPN)
    return p


# ---------------------------------------------------------------------------
# SDDS reading
# ---------------------------------------------------------------------------
class SddsReader:
    """Column / parameter access to SDDS files.

    ``backend="pysdds"`` uses the pysdds package; ``backend="text"`` runs
    ``sdds2stream`` from *exe_dir* (the directory of the elegant binary) and
    parses its tab-separated output.  ``None`` picks pysdds when importable.
    """

    def __init__(self, exe_dir: Path, backend: str | None = None):
        if backend is None:
            try:
                import pysdds  # noqa: F401
                backend = "pysdds"
            except ImportError:
                backend = "text"
        if backend not in ("pysdds", "text"):
            raise ValueError(f"unknown SDDS backend {backend!r} (pysdds, text)")
        self.backend = backend
        self.exe_dir = Path(exe_dir)
        if backend == "text":
            self._stream = self.exe_dir / "sdds2stream"
            if not self._stream.is_file():
                raise RuntimeError(f"sdds2stream not found next to elegant ({self.exe_dir}); "
                                   "install pysdds or the SDDS toolkit")

    # -- public ---------------------------------------------------------
    def n_pages(self, path: Path) -> int:
        if self.backend == "pysdds":
            return int(self._pysdds(path).n_pages)
        return int(self._run([str(path), "-npages=bare"]).strip().splitlines()[0])

    def columns(self, path: Path, names: list[str], page: int = 0,
                strings: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
        """Arrays for *names* on *page* (0-based).  With the text backend the
        columns listed in *strings* stay text, everything else is float."""
        if self.backend == "pysdds":
            f = self._pysdds(path)
            out = {}
            for n in names:
                v = f.col(n).data[page]
                out[n] = np.asarray(v) if n in strings else np.asarray(v, dtype=float)
            return out
        txt = self._run([str(path), "-columns=" + ",".join(names), f"-page={page + 1}",
                         "-delimiter=\t"])
        rows = [ln.split("\t") for ln in txt.splitlines() if ln.strip()]
        cols: dict[str, list[str]] = {n: [] for n in names}
        for r in rows:
            if len(r) != len(names):
                raise RuntimeError(f"sdds2stream: expected {len(names)} fields, got {len(r)}: "
                                   f"{r!r}")
            for n, v in zip(names, r, strict=True):
                cols[n].append(v.strip().strip('"'))
        return {n: (np.asarray(v) if n in strings else np.asarray(v, dtype=float))
                for n, v in cols.items()}

    def parameter(self, path: Path, name: str, page: int = 0) -> float:
        if self.backend == "pysdds":
            return float(self._pysdds(path).par(name).data[page])
        txt = self._run([str(path), f"-parameters={name}", f"-page={page + 1}"])
        return float(txt.strip().splitlines()[0].strip().strip('"'))

    # -- internals ------------------------------------------------------
    @staticmethod
    def _pysdds(path: Path):
        import pysdds

        return pysdds.read(str(path))

    def _run(self, args: list[str]) -> str:
        r = subprocess.run([str(self._stream), *args], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"sdds2stream {' '.join(args)} failed: "
                               f"{r.stderr.strip() or r.stdout.strip()}")
        return r.stdout


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _beta_gamma_from_pc(pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.sqrt(1.0 + pc * pc)
    return pc / g, g


def _kinetic_from_pc(pc: np.ndarray, mass_eV: float) -> np.ndarray:
    return (np.sqrt(1.0 + pc * pc) - 1.0) * mass_eV


def _change_particle(beam: BeamSpec) -> str:
    """``change_particle`` namelist for the BeamSpec species.

    Always ``name="custom"`` with the mass expressed in elegant's own electron
    mass, so the mass elegant uses equals ``SPECIES`` for every species — the
    built-in ``electron`` (m_e = 0.51099906 MeV, 2.2e-7 above CODATA-2018) and
    ``proton`` (1836.181 m_e, 1.5e-5 off) would leak into ``ref_kinetic_eV``
    and the cavity gains at the 1e-9 tier of PLAN §5.2."""
    mass_ratio = beam.mass_eV / ME_ELEGANT_EV
    charge_ratio = -beam.charge   # relative to the electron's charge (-e): electron +1, proton -1
    return (f'&change_particle name = "custom", mass_ratio = {mass_ratio:.15g}, '
            f'charge_ratio = {charge_ratio:d} &end')


def _align_rows(names: list[str], s_out: np.ndarray, tbl_names: np.ndarray, tbl_s: np.ndarray,
                what: str, tol: float = 1e-9) -> tuple[list[int], str | None]:
    """Row index into an auxiliary table (twiss / floor) for each matrix row.

    elegant writes the tables 1:1 with the matrix rows, so the identity is
    used when names and s agree; otherwise fall back to the cpymad rule (last
    row at a matching s wins) and say so."""
    if (len(tbl_names) == len(names)
            and all(str(a) == b for a, b in zip(tbl_names, names, strict=True))
            and np.all(np.abs(tbl_s - s_out) <= tol)):
        return list(range(len(names))), None
    idx = []
    for s in s_out:
        hits = np.nonzero(np.abs(tbl_s - s) <= tol)[0]
        if len(hits) == 0:
            raise ValueError(f"no {what} row at s={s!r}")
        idx.append(int(hits[-1]))
    return idx, (f"{what}: rows aligned by s (last row at a matching s wins); "
                 "row sequence differs from matrix_output")


def _write_probe_sdds(path: Path, native: np.ndarray, pc0: float) -> None:
    """Two-page ASCII SDDS beam file: page 1 the reference particle (fiducial),
    page 2 the probe.  Columns x xp y yp t p particleID."""
    beta0 = pc0 / np.sqrt(1.0 + pc0 * pc0)
    lines = [
        "SDDS1",
        "&description text=\"lattix probe bunch (page 1: reference particle, page 2: probe)\" &end",
        "&column name=x, units=m, type=double &end",
        "&column name=xp, type=double &end",
        "&column name=y, units=m, type=double &end",
        "&column name=yp, type=double &end",
        "&column name=t, units=s, type=double &end",
        "&column name=p, units=\"m$be$nc\", type=double &end",
        "&column name=particleID, type=long &end",
        "&data mode=ascii &end",
        "! page number 1",
        "1",
        f"0 0 0 0 0 {pc0:.17g} 1",
        "! page number 2",
        str(len(native)),
    ]
    for i, (x, xp, y, yp, s, d) in enumerate(native):
        t = s / (beta0 * C_LIGHT)          # elegant: s = β c t  (late-positive s -> later arrival)
        p = pc0 * (1.0 + d)                # δ = Δp/p0, p in units of m c (βγ)
        lines.append(f"{x:.17g} {xp:.17g} {y:.17g} {yp:.17g} {t:.17g} {p:.17g} {i + 1}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------
@register
class ElegantOracle:
    name = "elegant"
    formats = ("elegant",)
    basis = Basis.ELEGANT
    timeout_s = 600

    def __init__(self, exe: Path | str | None = None, sdds_backend: str | None = None):
        self._exe_override = Path(exe).expanduser() if exe else None
        self.sdds_backend = sdds_backend   # None: pysdds if importable, else text

    # ------------------------------------------------------------------
    def exe(self) -> Path:
        p = self._exe_override or find_elegant()
        if p is None:
            raise FileNotFoundError(
                "elegant binary not found: set ELEGANT_EXE or LATTIX_ENV_BIN, "
                f"or install it ({_DEFAULT_ENV_BIN}/elegant, PATH)")
        return p

    def available(self) -> tuple[bool, str]:
        try:
            exe = self.exe()
        except FileNotFoundError as e:
            return False, str(e)
        try:
            ver = elegant_version(str(exe))
        except Exception as e:  # noqa: BLE001
            return False, f"{exe} did not run: {e}"
        return True, f"{exe} elegant {ver}"

    def reader(self) -> SddsReader:
        return SddsReader(self.exe().parent, self.sdds_backend)

    # ------------------------------------------------------------------
    def run(self, deck: Path, *, fmt: str | None = None, beam: BeamSpec | None = None,
            probe: Probe | None = None, workdir: Path | None = None,
            use_beamline: str | None = None, matrix_mode: str = "individual",
            velocity_term: bool = True) -> OracleResult:
        """Run elegant on *deck* (an ``.lte`` file).

        ``matrix_mode="individual"`` (default) fills ``R_elem`` with per-element
        maps; ``"cumulative"`` with elegant's own start -> exit concatenations
        (``individual_matrices=0``), useful to cross-check ``R_cum``.
        """
        if fmt not in (None, "elegant"):
            raise ValueError(f"elegant reads .lte decks only, not {fmt!r}")
        if matrix_mode not in ("individual", "cumulative"):
            raise ValueError("matrix_mode must be 'individual' or 'cumulative', "
                             f"not {matrix_mode!r}")
        exe = self.exe()
        deck = Path(deck).resolve()
        if not deck.is_file():
            raise FileNotFoundError(deck)
        wd = Path(workdir).resolve() if workdir else Path(tempfile.mkdtemp(prefix="lattix_elegant_"))
        wd.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        if beam is None:
            beam = BeamSpec()
            warnings.append("elegant decks carry no beam definition; using BeamSpec() defaults "
                            f"({beam.species}, {beam.kinetic_energy_eV:.6g} eV)")
        mass_eV, charge = beam.mass_eV, beam.charge
        pc0 = beam.beta * beam.gamma

        defns = rpn_defns_path(wd)
        ele = self._command_file(deck, beam, pc0, use_beamline, matrix_mode, probe is not None)
        (wd / "run.ele").write_text(ele)
        native_probe = None
        if probe is not None:
            from lattix.oracles.basis import transform_matrix

            Tinv = np.linalg.inv(transform_matrix(Basis.ELEGANT, beam.kinetic_energy_eV, mass_eV))
            native_probe = np.asarray(probe.coords, dtype=float) @ Tinv.T
            _write_probe_sdds(wd / "probe.sdds", native_probe, pc0)

        log = self._execute(exe, wd, defns)
        warnings += self._warnings(log)
        rd = self.reader()

        # -- matrices ------------------------------------------------------
        mat = rd.columns(wd / "run.mat",
                         ["s", "pCentral", "ElementName", "ElementType", *_R_COLS],
                         strings=("ElementName", "ElementType"))
        all_names = [str(n) for n in mat["ElementName"]]
        keep = [i for i, n in enumerate(all_names) if n != _BEGIN_MARK]
        beg = [i for i, n in enumerate(all_names) if n == _BEGIN_MARK]
        names = [all_names[i] for i in keep]
        n = len(keep)
        if n == 0:
            raise RuntimeError("elegant matrix output has no element rows")
        R = np.empty((n, 6, 6))
        for a in range(6):
            for b in range(6):
                R[:, a, b] = mat[f"R{a + 1}{b + 1}"][keep]
        s_out = mat["s"][keep]
        s0 = float(mat["s"][beg[0]]) if beg else 0.0
        lengths = np.diff(np.concatenate([[s0], s_out]))
        pc_out = mat["pCentral"][keep]
        pc_start = float(mat["pCentral"][beg[0]]) if beg else pc0
        if abs(pc_start - pc0) > 1e-12 * max(pc0, 1.0):
            warnings.append(f"elegant start momentum βγ={pc_start!r} differs from BeamSpec {pc0!r}")
        pc_in = np.concatenate([[pc_start], pc_out[:-1]])
        ke_in = _kinetic_from_pc(pc_in, mass_eV)
        ke_out = _kinetic_from_pc(pc_out, mass_eV)
        # elegant's 5th coordinate is PATH LENGTH (its matrices carry no velocity-bunching
        # term: a drift has R56 = 0).  Every other engine and the common basis use an
        # arrival-time coordinate, where a particle with δ > 0 gains L·δ/γ² per element.
        # Add that term here (native s is late-positive, hence the minus sign) so the
        # per-element maps are comparable; γ is taken at the element entrance (exact for
        # non-accelerating elements, thin cavities have L = 0).  MEASURED 2026-09-03.
        # ``velocity_term=False`` returns elegant's raw path-length maps.
        if velocity_term:
            gamma_in = 1.0 + ke_in / mass_eV
            corr = lengths / (gamma_in * gamma_in)
            R[:, 4, 5] -= corr if matrix_mode == "individual" else np.cumsum(corr)

        # -- twiss / dispersion --------------------------------------------
        twi = rd.columns(wd / "run.twi", ["s", "ElementName", *_TWI_COLS.values()],
                         strings=("ElementName",))
        tkeep = [i for i, nm in enumerate(twi["ElementName"]) if str(nm) != _BEGIN_MARK]
        idx, note = _align_rows(names, s_out, twi["ElementName"][tkeep], twi["s"][tkeep],
                                "twiss_output")
        if note:
            warnings.append(note)
        sel = np.asarray(tkeep)[idx]
        twiss = {k: twi[c][sel] for k, c in _TWI_COLS.items()
                 if k in ("betx", "alfx", "bety", "alfy")}
        disp = {k: twi[c][sel] for k, c in _TWI_COLS.items() if k in ("dx", "dpx", "dy", "dpy")}

        # -- floor coordinates ---------------------------------------------
        flr = rd.columns(wd / "run.flr", ["s", "ElementName", *_FLR_COLS],
                         strings=("ElementName",))
        fkeep = [i for i, nm in enumerate(flr["ElementName"]) if str(nm) != _BEGIN_MARK]
        idx, note = _align_rows(names, s_out, flr["ElementName"][fkeep], flr["s"][fkeep],
                                "floor_coordinates")
        if note:
            warnings.append(note)
        sel = np.asarray(fkeep)[idx]
        survey = np.stack([flr[c][sel] for c in _FLR_COLS], axis=1)

        meta = {
            "workdir": str(wd), "command_file": str(wd / "run.ele"), "rpn_defns": str(defns),
            "elegant": str(exe), "elegant_version": elegant_version(str(exe)),
            "sdds_backend": rd.backend, "matrix_mode": matrix_mode, "velocity_term": velocity_term,
            "use_beamline": use_beamline, "p_central": pc0, "p_central_mev": beam.pc_eV * 1e-6,
            "change_particle": _change_particle(beam), "elegant_me_eV": ME_ELEGANT_EV,
            "longitudinal": ("s = path length (elegant s = βct); drift R56 = 0, "
                             "no L/γ² velocity term"),
        }

        # -- probe ---------------------------------------------------------
        probe_out = None
        if probe is not None:
            probe_out, raw = self._read_probe(rd, wd / "run.out", len(native_probe))
            meta["probe_raw"] = raw
            meta["probe_longitudinal"] = (
                "s_native = β_ref c (t - t_ref), late-positive arrival time; δ = p/p_ref - 1; "
                "reference particle tracked alone as the fiducial step 1")
            lost = int(np.isnan(probe_out[:, 0]).sum())
            if lost:
                warnings.append(f"{lost} probe particle(s) lost")

        return OracleResult(
            engine=self.name, basis=Basis.ELEGANT, names=names, length=lengths, s_out=s_out,
            R_elem=R, ref_kinetic_eV_in=ke_in, ref_kinetic_eV_out=ke_out, mass_eV=mass_eV,
            charge=charge, twiss=twiss, disp=disp, survey=survey, probe_out=probe_out,
            warnings=warnings, meta=meta,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _command_file(deck: Path, beam: BeamSpec, pc0: float, use_beamline: str | None,
                      matrix_mode: str, with_probe: bool) -> str:
        lines = ["! generated by lattix.oracles.elegant — do not edit",
                 f"! reference: {beam.species} K={beam.kinetic_energy_eV:.12g} eV, "
                 f"p={beam.pc_eV * 1e-6:.12g} MeV/c (= p_central_mev), βγ={pc0:.15g}"]
        lines.append(_change_particle(beam))
        rs = ["&run_setup", f'    lattice = "{deck}",']
        if use_beamline:
            rs.append(f'    use_beamline = "{use_beamline}",')
        rs += [f'    search_path = "{deck.parent}",',
               f"    p_central = {pc0:.15g},",
               "    default_order = 2,",
               "    always_change_p0 = 0,"]
        if with_probe:
            rs += ['    output = "%s.out",', '    centroid = "%s.cen",']
        rs.append("&end")
        lines += rs
        if with_probe:
            lines.append("&run_control n_steps = 2, first_is_fiducial = 1, "
                         "reset_rf_for_each_step = 0 &end")
        else:
            lines.append("&run_control n_steps = 1 &end")
        lines.append(
            '&twiss_output filename = "%s.twi", matched = 0, output_at_each_step = 0,\n'
            f"    beta_x = {beam.betx:.15g}, alpha_x = {beam.alfx:.15g}, "
            f"beta_y = {beam.bety:.15g}, alpha_y = {beam.alfy:.15g},\n"
            f"    eta_x = {beam.dx:.15g}, etap_x = {beam.dpx:.15g}, "
            f"eta_y = {beam.dy:.15g}, etap_y = {beam.dpy:.15g} &end")
        indiv = 1 if matrix_mode == "individual" else 0
        lines.append('&matrix_output SDDS_output = "%s.mat", SDDS_output_order = 1, '
                     f"individual_matrices = {indiv}, full_matrix_only = 0, "
                     "output_at_each_step = 0 &end")
        lines.append('&floor_coordinates filename = "%s.flr" &end')
        if with_probe:
            lines.append('&sdds_beam input = "probe.sdds", track_pages_separately = 1 &end')
            lines.append("&track &end")
        return "\n".join(lines) + "\n"

    def _execute(self, exe: Path, wd: Path, defns: Path) -> str:
        cmd = [str(exe), "run.ele", f"-rpnDefns={defns}"]
        try:
            r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True,
                               timeout=self.timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"elegant timed out after {self.timeout_s} s in {wd}") from e
        log = r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr.strip() else "")
        (wd / "run.log").write_text(log)
        missing = [f for f in ("run.mat", "run.twi", "run.flr") if not (wd / f).is_file()]
        if r.returncode != 0 or missing:
            tail = "\n".join(log.strip().splitlines()[-30:])
            what = f"exit status {r.returncode}" if r.returncode else f"missing output {missing}"
            raise RuntimeError(f"elegant failed ({what}) in {wd}:\n{tail}")
        return log

    @staticmethod
    def _warnings(log: str) -> list[str]:
        out: list[str] = []
        for ln in log.splitlines():
            ln = ln.strip()
            if ln.startswith("*** Warning") and ln not in out:
                out.append(ln)
        return out

    @staticmethod
    def _read_probe(rd: SddsReader, out: Path, n: int) -> tuple[np.ndarray, dict]:
        if not out.is_file():
            raise RuntimeError(f"elegant wrote no tracking output {out}")
        if rd.n_pages(out) != 2:
            raise RuntimeError(f"expected 2 pages (fiducial + probe) in {out}, "
                               f"got {rd.n_pages(out)}")
        ref = rd.columns(out, ["t", "p"], page=0)
        if len(ref["t"]) != 1:
            raise RuntimeError("fiducial page must hold exactly the reference particle")
        t_ref, p_ref = float(ref["t"][0]), float(ref["p"][0])
        pc_end = rd.parameter(out, "pCentral", page=1)
        cols = rd.columns(out, ["x", "xp", "y", "yp", "t", "p", "particleID"], page=1)
        ids = np.rint(cols["particleID"]).astype(int) - 1
        if np.any(ids < 0) or np.any(ids >= n):
            raise RuntimeError("unexpected particleID in elegant output")
        beta_ref = p_ref / np.sqrt(1.0 + p_ref * p_ref)
        res = np.full((n, 6), np.nan)
        for k, c in enumerate(("x", "xp", "y", "yp")):
            res[ids, k] = cols[c]
        res[ids, 4] = beta_ref * C_LIGHT * (cols["t"] - t_ref)   # late-positive arrival time
        res[ids, 5] = cols["p"] / p_ref - 1.0
        t_raw = np.full(n, np.nan)
        p_raw = np.full(n, np.nan)
        t_raw[ids] = cols["t"]
        p_raw[ids] = cols["p"]
        raw = {"t": t_raw, "p": p_raw, "t_ref": t_ref, "p_ref": p_ref, "pCentral_end": pc_end}
        return res, raw
