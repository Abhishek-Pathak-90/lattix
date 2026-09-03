"""FLAME oracle tests (PLAN §6 task 3.4) — everything here runs the real engine.

Run under an interpreter that can ``import flame``::

    PYTHONPATH=. /path/to/envs/lattix/bin/python -m pytest tests/oracles/test_flame_adapter.py

``flame-code`` publishes **manylinux x86_64 wheels only** (no macOS wheel, no sdist), so
on this machine FLAME 1.9.2 was built from source into the ``lattix`` conda env; the
adapter finds any interpreter that can import it (``LATTIX_FLAME_PYTHON`` /
``LATTIX_FLAME_ENV``) and otherwise every test here skips.

What is asserted:

* the **basis fingerprint** — a 1 m drift in the common basis is ``R56 = +L/γ²``,
  which is what ``basis._Z_SIGN[Basis.FLAME] = −1`` (φ late-positive) produces.  The
  same number is derivable from ``src/moment.cpp:1119``
  (``transfer(PS_S, PS_PS) = −2π·L[mm]/(λ[mm]·IonEs[MeV/u]·(βγ)³)``), so the test
  pins engine and source against each other;
* the **sign conventions** measured, never recalled: ``orbtrim theta_x`` is a +x
  deflection, ``quadrupole B2 > 0`` focuses x, and ``roll`` *is* MAD-X's ``tilt``;
* the **rfcavity phase convention** — ``syncflag`` 1/2 make ``phi`` a synchronous
  phase (crest near 0, cos convention), ``syncflag = 0`` a driven phase;
* **FLAME accepts every deck the writer produces** (goldens and the vendored corpus);
* **round-trip lockstep** — the vendored FRIB decks read → written → propagated give
  identical reference energies and per-element maps to 1e-9;
* **FODO cross-engine** — ``fodo.madx`` → IR → GLPS agrees with cpymad on the source
  deck to 1e-8 after the basis transform.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

import lattix.oracles.flame as flame_adapter  # noqa: F401  (registers the "flame" oracle)
from lattix.formats.flame import Reader, Writer
from lattix.oracles import BeamSpec, get_oracle
from lattix.oracles.base import Basis
from lattix.oracles.basis import drift_common
from lattix.oracles.compare import compare_pair
from lattix.oracles.flame import cavity_data_dir
from lattix.testing import needs

pytestmark = [pytest.mark.oracle_flame, needs("flame")]

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "flame"
HELIX = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "flame"

M_P = 938272088.16
KE = 2.1e6
FREQ = 162.5e6

BEAM_HEADER = f"""sim_type = "MomentMatrix";
IonEs = {M_P!r};
IonEk = {KE!r};
IonZ = 1.0;
IonChargeStates = [1.0];
NCharge = [1.0];
SampleFreq = {FREQ!r};
BaryCenter0 = [0, 0, 0, 0, 0, 0, 1];
S0 = [1,0,0,0,0,0,0, 0,1e-6,0,0,0,0,0, 0,0,1,0,0,0,0, 0,0,0,1e-6,0,0,0,
      0,0,0,0,1,0,0, 0,0,0,0,0,1e-6,0, 0,0,0,0,0,0,0];
S: source, vector_variable = "BaryCenter", matrix_variable = "S";
"""


@pytest.fixture(scope="module")
def oracle():
    return get_oracle("flame")


def make_deck(tmp_path: Path, body: str, name: str = "d.lat") -> Path:
    """A one-line proton deck: ``body`` defines the elements, ``cell`` uses them."""
    names = [ln.split(":")[0].strip() for ln in body.strip().splitlines()
             if ":" in ln and not ln.strip().startswith("#")]
    text = BEAM_HEADER + body + f"\ncell: LINE = (S, {', '.join(names)});\nUSE: cell;\n"
    p = tmp_path / name
    p.write_text(text)
    return p


def workdir_with_cavity_data(tmp_path: Path) -> Path:
    """A directory where ``Eng_Data_Dir = dir("data")`` resolves."""
    wd = tmp_path / "run"
    wd.mkdir(exist_ok=True)
    link = wd / "data"
    if not link.exists():
        link.symlink_to(cavity_data_dir())
    return wd


# --------------------------------------------------------------- availability
def test_oracle_registers_and_reports_itself(oracle):
    assert oracle.name == "flame"
    assert oracle.formats == ("flame",)
    ok, why = oracle.available()
    assert ok, why


# ------------------------------------------------------------ basis fingerprint
def test_drift_fingerprint_matches_the_common_basis(tmp_path, oracle):
    """1 m drift → ``R56 = +L/γ²`` in the common basis (PLAN §5.1 fingerprint rule).

    This is the whole content of ``basis._Z_SIGN[Basis.FLAME] = −1``: FLAME's φ is
    *late-positive*, so a higher-energy particle arrives at smaller φ.
    """
    deck = make_deck(tmp_path, "D1: drift, L = 1.0;")
    res = oracle.run(deck)
    assert res.basis is Basis.FLAME
    assert res.names == ["S", "D1"]
    assert res.meta["types"] == ["source", "drift"]
    assert res.length == pytest.approx([0.0, 1.0])

    native = res.R_elem[1]
    assert native[0, 1] == pytest.approx(1000.0)      # x' [rad] -> x [mm]: 1 m = 1000 mm
    assert native[2, 3] == pytest.approx(1000.0)
    assert native[4, 5] < 0                           # phi is late-positive

    # the source formula, moment.cpp:1119 / moment_sup.cpp
    gamma = 1 + KE / M_P
    bg = math.sqrt(gamma * gamma - 1)
    lam_mm = 299792458.0 / FREQ * 1e3
    expected = -2 * math.pi * 1000.0 / (lam_mm * (M_P / 1e6) * bg**3)
    assert native[4, 5] == pytest.approx(expected, rel=1e-12)

    common = res.to_common().R_elem[1]
    assert common == pytest.approx(drift_common(1.0, KE, M_P), abs=1e-13)
    assert common[4, 5] == pytest.approx(1.0 / gamma**2, rel=1e-12)


def test_reference_energy_is_per_nucleon(tmp_path, oracle):
    """FLAME's IonEs/IonEk are eV/u; the adapter reports them unscaled so that
    ``basis.transform_matrix``'s ΔE_k [MeV/u] → δ row is right."""
    deck = make_deck(tmp_path, "D1: drift, L = 1.0;")
    res = oracle.run(deck)
    assert res.mass_eV == pytest.approx(M_P)
    assert res.ref_kinetic_eV_in == pytest.approx([KE, KE])
    assert res.meta["mass_number"] == 1
    assert "PER NUCLEON" in res.meta["units"]
    assert res.rf_frequency_Hz == pytest.approx([FREQ, FREQ])


def test_uranium_deck_reports_the_charge_state(oracle, tmp_path):
    wd = workdir_with_cavity_data(tmp_path)
    lat, _ = Reader().read(DATA / "LS1FS1_lattice.lat")
    out = wd / "u.lat"
    Writer().write(lat, out, eng_data_dir="data")
    res = oracle.run(out)
    assert res.charge == 33
    assert res.meta["mass_number"] == 238
    assert res.mass_eV == pytest.approx(931.49432e6)
    assert len(res.meta["charge_states"]) == 2


# --------------------------------------------------------------- sign conventions
def test_orbtrim_theta_x_deflects_towards_plus_x(tmp_path, oracle):
    """``moment.cpp:897``: ``transfer(PS_PX, 6) = +theta_x`` — the IR/MAD-X hkick sign."""
    deck = make_deck(tmp_path, "K1: orbtrim, theta_x = 0.001, theta_y = 0.002;")
    res = oracle.run(deck)
    col = res.meta["dipole_column"][1]
    assert col[1] == pytest.approx(0.001)
    assert col[3] == pytest.approx(0.002)
    assert np.allclose(res.R_elem[1], np.eye(6))


@needs("madx")
def test_quadrupole_B2_matches_madx_k1(tmp_path, oracle):
    """``B2`` is a lab gradient [T/m] and ``k1 = B2/Bρ`` with the MAD-X sign."""
    brho = math.sqrt((KE + M_P) ** 2 - M_P**2) / 299792458.0
    k1 = 1.25
    deck = make_deck(tmp_path, f"Q1: quadrupole, L = 0.3, B2 = {k1 * brho!r};")
    f = oracle.run(deck).to_common()

    madx = tmp_path / "q.madx"
    madx.write_text(f"""
beam, particle=proton, energy={(KE + M_P) / 1e9!r};
Q1: quadrupole, l=0.3, k1={k1!r};
seq: sequence, l=0.3;
  Q1, at=0.15;
endsequence;
use, sequence=seq;
""")
    m = get_oracle("madx").run(madx, beam=BeamSpec(species="proton",
                                                   kinetic_energy_eV=KE)).to_common()
    assert f.R_elem[1][:4, :4] == pytest.approx(m.R_elem[-1][:4, :4], abs=1e-10)
    assert f.R_elem[1][1, 0] < 0            # B2 > 0 focuses x


@needs("madx")
def test_roll_is_the_madx_tilt(tmp_path, oracle):
    """The skew→``roll`` sign the writer uses, measured rather than recalled: a FLAME
    quadrupole rolled by +π/8 is MAD-X's ``(k1, k1s) = k·(cos π/4, −sin π/4)``."""
    brho = math.sqrt((KE + M_P) ** 2 - M_P**2) / 299792458.0
    roll = math.pi / 8
    deck = make_deck(tmp_path, f"Q1: quadrupole, L = 0.3, B2 = {brho!r}, roll = {roll!r};")
    f = oracle.run(deck).to_common()

    madx = tmp_path / "qs.madx"
    madx.write_text(f"""
beam, particle=proton, energy={(KE + M_P) / 1e9!r};
Q1: quadrupole, l=0.3, k1={math.cos(2 * roll)!r}, k1s={-math.sin(2 * roll)!r};
seq: sequence, l=0.3;
  Q1, at=0.15;
endsequence;
use, sequence=seq;
""")
    m = get_oracle("madx").run(madx, beam=BeamSpec(species="proton",
                                                   kinetic_energy_eV=KE)).to_common()
    assert f.R_elem[1][:4, :4] == pytest.approx(m.R_elem[-1][:4, :4], abs=1e-9)
    assert abs(f.R_elem[1][1, 2]) > 1e-3     # the coupling is real, not a null test


# ------------------------------------------------------ rfcavity phase convention
def _cavity_gain(tmp_path, oracle, phi_deg: float, syncflag=None,
                 cavtype="0.041QWR", scl=0.64, ek=0.5e6) -> float:
    """Reference energy gain [eV/u] of one FRIB QWR at the given deck phase."""
    wd = workdir_with_cavity_data(tmp_path)
    sf = f"syncflag = {syncflag}, " if syncflag is not None else ""
    text = f"""sim_type = "MomentMatrix";
MpoleLevel = "2";
IonEs = 931494320.0; IonEk = {ek!r}; IonZ = 0.13865546218487396;
IonChargeStates = [0.13865546218487396]; NCharge = [1.0];
Eng_Data_Dir = dir("data");
BaryCenter0 = [0,0,0,0,0,0,1];
S0 = [1,0,0,0,0,0,0, 0,1e-6,0,0,0,0,0, 0,0,1,0,0,0,0, 0,0,0,1e-6,0,0,0,
      0,0,0,0,1,0,0, 0,0,0,0,0,1e-6,0, 0,0,0,0,0,0,0];
S: source, vector_variable = "BaryCenter", matrix_variable = "S";
C: rfcavity, cavtype = "{cavtype}", L = 0.24, f = 80.5e6, phi = {phi_deg!r}, {sf}scl_fac = {scl!r};
cell: LINE = (S, C);
USE: cell;
"""
    deck = wd / f"cav_{phi_deg:+.3f}_{syncflag}.lat"
    deck.write_text(text)
    res = oracle.run(deck)
    return float(res.ref_kinetic_eV_out[-1] - res.ref_kinetic_eV_in[0])


@pytest.mark.parametrize("syncflag", [None, 2])
def test_synchronous_phase_crests_near_zero(tmp_path, oracle, syncflag):
    """``syncflag`` 1 (default) and 2 make ``phi`` a *synchronous* phase, so 0 is (almost)
    crest and the IR mapping ``phase_rad = radians(phi)`` needs no shift.

    Measured 2026-09-03 (0.041QWR, 0.5 MeV/u, scl_fac 0.64): the maximum sits at
    ``phi = +3.7°`` and the gain follows ``cos φ`` only to ~3 % — FLAME's cavity is a
    tabulated multi-gap thin-lens model, which is exactly why the reader records
    EQUIVALENT ``FLAME_CAVTYPE_VOLTAGE_UNKNOWN`` instead of inventing a voltage.
    """
    scan = np.arange(-180.0, 180.0, 5.0)
    gains = np.array([_cavity_gain(tmp_path, oracle, float(p), syncflag) for p in scan])
    crest = scan[int(np.argmax(gains))]
    assert abs(crest) <= 10.0, crest
    peak = gains.max()
    assert peak == pytest.approx(67479.0, rel=1e-3)
    assert _cavity_gain(tmp_path, oracle, 0.0, syncflag) / peak > 0.99
    # a cosine model would give 0.819 at -35 deg; FLAME's tabulated model gives 0.793
    ratio = _cavity_gain(tmp_path, oracle, -35.0, syncflag) / peak
    assert ratio == pytest.approx(0.7933, abs=2e-3)
    assert abs(ratio - math.cos(math.radians(35.0))) > 0.02


def test_driven_phase_is_offset_from_the_synchronous_one(tmp_path, oracle):
    """``syncflag = 0`` makes ``phi`` the *driven* phase: the same peak gain, but ~84°
    away.  The reader records EQUIVALENT ``FLAME_DRIVEN_PHASE`` and sets
    ``RFP.phase_is_sync = False`` so no downstream writer treats it as a crest phase."""
    scan = np.arange(-180.0, 180.0, 5.0)
    gains = np.array([_cavity_gain(tmp_path, oracle, float(p), 0) for p in scan])
    crest = scan[int(np.argmax(gains))]
    assert crest == pytest.approx(-80.0, abs=6.0)
    assert gains.max() == pytest.approx(67479.0, rel=1e-3)
    assert _cavity_gain(tmp_path, oracle, 90.0, 0) < 0        # decelerating


# ---------------------------------------------- FLAME accepts what the writer writes
def _runnable_golden(golden: str, tmp_path: Path) -> tuple[Path, Path]:
    """Copy a golden into a work directory FLAME can construct it from."""
    wd = workdir_with_cavity_data(tmp_path)
    deck = wd / golden
    text = (GOLDEN / golden).read_text()
    # a golden with an rfcavity but no Eng_Data_Dir is a *correct* writer output (the
    # ledger says LOSSY FLAME_NO_ENG_DATA_DIR); FLAME needs the tables to construct it
    if "rfcavity" in text and "Eng_Data_Dir" not in text:
        text = text.replace('sim_type = "MomentMatrix";',
                            'sim_type = "MomentMatrix";\n'
                            f'Eng_Data_Dir = dir("{cavity_data_dir()}");')
    deck.write_text(text)
    return wd, deck


@pytest.mark.parametrize("golden", sorted(p.name for p in GOLDEN.glob("*.lat")))
def test_flame_machine_runs_every_golden(golden, tmp_path, oracle):
    """Every committed writer snapshot builds and propagates in FLAME."""
    _, deck = _runnable_golden(golden, tmp_path)
    res = oracle.run(deck)
    assert res.meta["types"][0] == "source"
    assert np.isfinite(res.R_elem).all()


@pytest.mark.parametrize("golden", sorted(p.name for p in GOLDEN.glob("*.lat")))
def test_flame_glps_parser_accepts_every_golden(golden, tmp_path):
    """FLAME's own ``GLPSParser``/``Machine``, in-process (skipped when the running
    interpreter cannot import flame — the oracle then goes through its worker)."""
    pytest.importorskip("flame", reason="flame is not importable in this interpreter")
    from flame import GLPSParser, Machine

    wd = workdir_with_cavity_data(tmp_path)
    _, deck = _runnable_golden(golden, tmp_path)
    # GLPSParser.parse returns the raw (key, value) list, Machine.conf() an OrderedDict
    conf = dict(GLPSParser().parse(deck.read_bytes(), path=str(wd)))
    assert conf["sim_type"] == "MomentMatrix"
    machine = Machine(deck.read_bytes(), path=str(wd))
    assert len(machine) >= 2
    assert machine.conf()["elements"][0]["type"] == "source"


@pytest.mark.parametrize("name", sorted(p.name for p in DATA.glob("*.lat")))
def test_flame_runs_every_written_vendored_deck(name, tmp_path, oracle):
    """Every vendored deck read by lattix and written back propagates in FLAME —
    including ``FrontEnd.lat``, which FLAME *rejects* as shipped (it has ``ver = v``
    with ``v`` undefined) and which lattix repairs by dropping the bad attribute."""
    wd = workdir_with_cavity_data(tmp_path)
    lat, _ = Reader().read(DATA / name)
    out = wd / name
    Writer().write(lat, out, eng_data_dir="data")
    res = oracle.run(out)
    assert res.meta["types"][0] == "source"
    # +1 only when the deck had no `source` of its own and the writer prepended one
    assert res.n - len(lat.flatten()) in (0, 1)
    assert np.isfinite(res.R_elem).all()


def test_front_end_deck_is_broken_as_shipped(tmp_path, oracle):
    wd = workdir_with_cavity_data(tmp_path)
    deck = wd / "FrontEnd.lat"
    shutil.copy(DATA / "FrontEnd.lat", deck)
    with pytest.raises(Exception, match="v"):
        oracle.run(deck)


# ------------------------------------------------------------- round-trip lockstep
@pytest.mark.parametrize("name", ["LS1FS1_lattice.lat", "LS1.lat", "ALL_lattice.lat",
                                  "TMtest.lat"])
def test_read_write_lockstep(name, tmp_path, oracle):
    """FLAME propagates the vendored deck and lattix's re-write identically: the same
    per-element ``ref_IonEk`` and ``transmat`` to 1e-9 (PLAN §6 Phase 3 gate).

    Measured 2026-09-03 the difference is **exactly zero** on all four decks (2549
    elements for ``ALL_lattice.lat``) because :func:`~lattix.formats.flame.writer.fmt`
    is round-trip exact; the 1e-9 gate is what a FLAME upgrade must still meet."""
    wd = workdir_with_cavity_data(tmp_path)
    original = wd / "original.lat"
    shutil.copy(DATA / name, original)
    lat, _ = Reader().read(DATA / name)
    written = wd / "written.lat"
    Writer().write(lat, written, eng_data_dir="data")

    a = oracle.run(original)
    b = oracle.run(written)
    assert a.n == b.n
    assert a.meta["types"] == b.meta["types"]
    assert a.s_out == pytest.approx(b.s_out, abs=1e-12)
    assert a.ref_kinetic_eV_out == pytest.approx(b.ref_kinetic_eV_out, rel=1e-12, abs=1e-9)
    assert np.abs(a.R_elem - b.R_elem).max() < 1e-9


def test_ls1fs1_reference_energy_gain(tmp_path, oracle):
    """The FRIB LS1+FS1 line accelerates ²³⁸U³³⁺ from 0.5 to 16.82 MeV/u; the number is
    pinned so a reader or writer regression that silently changes a cavity shows up."""
    wd = workdir_with_cavity_data(tmp_path)
    lat, _ = Reader().read(DATA / "LS1FS1_lattice.lat")
    out = wd / "w.lat"
    Writer().write(lat, out, eng_data_dir="data")
    res = oracle.run(out)
    assert res.ref_kinetic_eV_in[0] == pytest.approx(0.5e6)
    assert res.ref_kinetic_eV_out[-1] == pytest.approx(16816951.191958785, rel=1e-9)
    assert res.s_out[-1] == pytest.approx(158.093655, abs=1e-9)


# ---------------------------------------------------------------- cross-engine FODO
@needs("madx")
def test_fodo_madx_to_flame_matches_cpymad(tmp_path, oracle):
    """``fodo.madx`` → IR → GLPS, then FLAME vs cpymad *on the original deck*: two
    independent engines, one on each end (PLAN §5.2 anti-cancellation rule).

    FLAME's ``sbend`` uses the same linear sector-bend + pole-face model as MAD-X
    (``moment_sup.cpp:228``: ``Kx = K + 1/ρ²``, ``Ky = −K``, edge matrices from
    ``tan(φ1)/ρ``), and ``HdipoleFitMode`` defaults to 1 so ``phi`` is the geometric
    angle — hence Exact tier, not Equivalent.
    """
    from lattix.formats.madx import Reader as MadxReader

    src = HELIX / "fodo.madx"
    lat, _ = MadxReader().read(src)
    out = tmp_path / "fodo.lat"
    Writer().write(lat, out)

    f = oracle.run(out)
    m = get_oracle("madx").run(src, beam=BeamSpec(
        species="proton", kinetic_energy_eV=lat.reference.kinetic_energy_eV,
        frequency_Hz=80.5e6))

    cmp = compare_pair(f, m)
    assert cmp.n_shared >= 8
    assert cmp.length_a == pytest.approx(6.6)
    assert cmp.length_b == pytest.approx(6.6)
    assert cmp.max_rcum_abs < 1e-8, cmp.row()
    for block, value in cmp.blocks.items():
        assert value < 1e-8, f"{block}: {value:.3e}\n{cmp.row()}"


@needs("madx")
def test_fodo_per_element_maps(tmp_path, oracle):
    """The same FODO element by element (not just cumulative), matched by name."""
    from lattix.formats.madx import Reader as MadxReader

    src = HELIX / "fodo.madx"
    lat, _ = MadxReader().read(src)
    out = tmp_path / "fodo.lat"
    Writer().write(lat, out)
    f = oracle.run(out).to_common()
    m = get_oracle("madx").run(src, beam=BeamSpec(
        species="proton", kinetic_energy_eV=lat.reference.kinetic_energy_eV)).to_common()

    worst = 0.0
    compared = 0
    for i, name in enumerate(f.names):
        if name.lower() not in [n.lower() for n in m.names]:
            continue
        j = [n.lower() for n in m.names].index(name.lower())
        if f.length[i] == 0.0 and m.length[j] == 0.0:
            continue
        assert f.length[i] == pytest.approx(m.length[j], abs=1e-12), name
        worst = max(worst, float(np.abs(f.R_elem[i] - m.R_elem[j]).max()))
        compared += 1
    assert compared >= 6
    assert worst < 1e-8, worst
