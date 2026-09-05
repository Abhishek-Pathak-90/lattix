"""PyORBIT3 as an engine (Phase 5.4): TraceWin decks against HELIX, the public SNS/ESS decks through
PyORBIT before and after a lattix rewrite, and the negative-charge gap convention.  Runs where PyORBIT3
is installed (conda env ``pyorbit`` or ``LATTIX_PYORBIT_PYTHON``; marker ``oracle_pyorbit``)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import _beam
from lattix.oracles import get_oracle
from lattix.oracles.base import BeamSpec
from lattix.oracles.compare import compare_pair
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, V_VOLT

pytestmark = pytest.mark.oracle_pyorbit
DATA = Path(__file__).resolve().parents[1] / "data" / "public"


@pytest.fixture(scope="module")
def pyorbit():
    o = get_oracle("pyorbit")
    ok, why = o.available()
    if not ok:
        pytest.skip(f"PyORBIT3 unavailable: {why}")
    return o


@pytest.mark.oracle_helix
@pytest.mark.parametrize("deck, sp, tol", [("fodo_cell.dat", "proton", 1e-8), ("solenoid_channel.dat", "proton", 1e-8),
                                            ("bend_line.dat", "proton", 1e-8), ("mebt_line.dat", "h-", 2e-2)])
def test_tracewin_decks_against_helix(pyorbit, tmp_path, deck, sp, tol):
    """Quads, solenoids (B = B₀/Bρ in 1/m, charge in the tracker) and sector bends exact; the MEBT's
    thin gaps carry PyORBIT's own gap focusing (BaseRfGap) — Equivalent tier (measured 4.3e-3),
    the reference energy exact (one <Cavity> per gap)."""
    helix = get_oracle("helix")
    if not helix.available()[0]:
        pytest.skip("HELIX not available")
    lat, _ = read(DATA / "helix" / deck, "tracewin", species=sp, kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    out = tmp_path / (Path(deck).stem + ".pyorbit.xml")
    write(lat, out, "pyorbit")
    rp = pyorbit.run(out, fmt="pyorbit", beam=_beam(lat), workdir=tmp_path / "po")
    rh = helix.run(DATA / "helix" / deck, fmt="tracewin", beam=_beam(lat), workdir=tmp_path / "hx")
    pc = compare_pair(rh, rp)
    assert pc.n_shared >= 7
    assert pc.blocks["T4x4"] < tol and pc.blocks["disp"] < tol
    assert pc.energy_rel < 1e-9


@pytest.mark.parametrize("name, sp, ke, freq", [("sns_mebt.xml", "h-", 2.5e6, 402.5e6),
                                                 ("ess_mebt.xml", "proton", 3.6e6, 352.21e6)])
def test_public_decks_survive_a_lattix_rewrite(pyorbit, tmp_path, name, sp, ke, freq):
    """PyORBIT on the original SNS/ESS MEBT and on lattix's read → write of it: the same maps
    (5e-9 / 3e-10) and energies — the thin correctors inside quads, the per-gap cavities and the
    TTF passthrough all survive."""
    deck = DATA / "pyorbit3" / name
    lat, _ = read(deck, species=sp, kinetic_energy_eV=ke)
    out = tmp_path / name
    write(lat, out, "pyorbit")
    beam = BeamSpec(species=sp, kinetic_energy_eV=ke, frequency_Hz=freq)
    r1 = pyorbit.run(deck, fmt="pyorbit", beam=beam, workdir=tmp_path / "orig")
    r2 = pyorbit.run(out, fmt="pyorbit", beam=beam, workdir=tmp_path / "rw")
    pc = compare_pair(r1, r2)
    assert pc.n_shared >= 30
    assert pc.blocks["T4x4"] < 1e-8 and pc.blocks["disp"] < 1e-8 and pc.blocks["R56"] < 1e-8
    assert pc.energy_rel < 1e-9
    assert r1.ref_kinetic_eV_out[-1] == pytest.approx(r2.ref_kinetic_eV_out[-1], rel=1e-9)


def test_negative_charge_gains_at_the_same_phase(pyorbit, tmp_path):
    """ΔE = q·E0TL·cos(phase): the writer's +180° for a negative particle keeps the IR's V·cos φ."""
    from lattix.ir.elements import RFP, Drift, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    for sp in ("proton", "h-"):
        ref = ReferenceParticle(species=species(sp), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
        lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                          RFCavity(name="c", length=0.0,
                                                   rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0),
                                                          frequency_Hz=FREQ_HZ)),
                                          Drift(name="d2", length=0.5)], ref)
        out = tmp_path / f"{sp}.pyorbit.xml"
        write(lat, out, "pyorbit")
        r = pyorbit.run(out, fmt="pyorbit", beam=_beam(lat), workdir=tmp_path / sp)
        k = next(i for i, n in enumerate(r.names) if n.startswith("c"))
        gain = r.ref_kinetic_eV_out[k] - r.ref_kinetic_eV_in[k]
        assert gain == pytest.approx(V_VOLT * math.cos(math.radians(30.0)), rel=1e-9)
        assert r.to_common().R_elem[k][5, 4] < 0                       # bunching at φs = −30°
