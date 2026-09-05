"""IMPACT-T adapter (conda-forge impact-t 3.1.5, ``ImpactTexe``): decks written by lattix against HELIX,
MAD-X and the analytic solenoid; the dipole model is report-only."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix import read, write
from lattix.crossval import _beam
from lattix.ir.elements import RFP, Drift, RFCavity, Solenoid, SolenoidP
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.oracles import get_oracle
from lattix.oracles.compare import compare_pair
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, V_VOLT

pytestmark = pytest.mark.oracle_impactt
DATA = Path(__file__).resolve().parents[1] / "data" / "public"


@pytest.fixture(scope="module")
def impactt():
    o = get_oracle("impactt")
    ok, why = o.available()
    if not ok:
        pytest.skip(f"IMPACT-T unavailable: {why}")
    return o


def _write(lat, tmp_path, name):
    out = tmp_path / name / "ImpactT.in"
    write(lat, out, "impactt")
    return out


@pytest.mark.oracle_helix
@pytest.mark.parametrize("deck, sp, tol, e_tol", [("fodo_cell.dat", "proton", 1e-7, 1e-9),
                                                   ("solenoid_channel.dat", "proton", 5e-6, 1e-9),
                                                   ("mebt_line.dat", "h-", 2e-2, 1e-5),
                                                   ("bend_line.dat", "proton", None, 1e-9)])
def test_tracewin_decks_against_helix(impactt, tmp_path, deck, sp, tol, e_tol):
    """Quads and drifts exact (2.3e-8), the solenoid table 9e-7 (MEASURED, docs/oracles.md), the MEBT's
    1 mm profile gaps Equivalent (6.4e-3 transverse, energy 1.4e-6); IMPACT-T's dipole bends the whole
    bunch by the reference angle (no pole-face focusing), so bend decks are report-only."""
    helix = get_oracle("helix")
    if not helix.available()[0]:
        pytest.skip("HELIX not available")
    lat, _ = read(DATA / "helix" / deck, "tracewin", species=sp, kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    out = _write(lat, tmp_path, Path(deck).stem)
    rt = impactt.run(out, fmt="impactt", beam=_beam(lat), workdir=tmp_path / "it")
    rh = helix.run(DATA / "helix" / deck, fmt="tracewin", beam=_beam(lat), workdir=tmp_path / "hx")
    pc = compare_pair(rh, rt)
    assert pc.n_shared >= 7
    assert pc.energy_rel < e_tol
    if tol is None:
        assert rt.meta["bends"] and rt.meta["bend_model"]
        assert pc.blocks["T4x4"] > 1e-3                             # the model difference is real
    else:
        assert pc.blocks["T4x4"] < tol and pc.blocks["disp"] < max(tol, 1e-9)


@pytest.mark.oracle_madx
def test_fodo_madx_quads_and_drifts_against_cpymad(impactt, tmp_path):
    madx = get_oracle("madx")
    if not madx.available()[0]:
        pytest.skip("cpymad not available")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = _write(lat, tmp_path, "fodo")
    rt = impactt.run(out, fmt="impactt", beam=_beam(lat), workdir=tmp_path / "it")
    rm = madx.run(DATA / "helix" / "fodo.madx", fmt="madx", beam=_beam(lat), workdir=tmp_path / "mx").to_common()
    assert rt.n == rm.n
    for i, name in enumerate(rt.names):
        d = np.abs(rt.R_elem[i][:4, :4] - rm.R_elem[i][:4, :4]).max()
        if name.startswith("bend"):
            assert 1e-3 < d < 5e-2, (name, d)                          # IMPACT-T's dipole model
        else:
            assert d < 1e-8, (name, d)


def test_solenoid_table_matches_the_hard_edge_map(impactt, tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)
    lat = Lattice.from_sequence("s", [Drift(name="d1", length=0.3),
                                      Solenoid(name="sol", length=0.4, solenoid=SolenoidP(Bsol_T=0.5)),
                                      Drift(name="d2", length=0.3)], ref)
    out = _write(lat, tmp_path, "sol")
    r = impactt.run(out, fmt="impactt", beam=_beam(lat), workdir=tmp_path / "it")
    K = 0.5 / (2 * ref.brho_signed)
    C, S = math.cos(K * 0.4), math.sin(K * 0.4)
    A = np.array([[C * C, S * C / K, S * C, S * S / K], [-K * S * C, C * C, -K * S * S, S * C],
                  [-S * C, -S * S / K, C * C, S * C / K], [K * S * S, -S * C, -K * S * C, C * C]])
    k = next(i for i, n in enumerate(r.names) if n.startswith("solenoid"))
    assert np.abs(r.R_elem[k][:4, :4] - A).max() < 5e-6                # MEASURED 9e-7 at a 1 ps step
    drift = np.array([[1, 0.3, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1]])
    for i in (k - 1, k + 1):
        assert np.abs(r.R_elem[i][:4, :4] - drift).max() < 1e-9


@pytest.mark.parametrize("sp", ["proton", "h-"])
def test_thin_gap_gain_and_phase_slope(impactt, tmp_path, sp):
    """A thin gap becomes a 1 mm raised-cosine profile (type 104) with a driven phase from lattix's
    time of flight: the reference gains V·cos φs for both charge signs and the slope bunches."""
    ref = ReferenceParticle(species=species(sp), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", length=0.0,
                                               rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0),
                                                      frequency_Hz=FREQ_HZ)),
                                      Drift(name="d2", length=0.5)], ref)
    out = _write(lat, tmp_path, sp)
    r = impactt.run(out, fmt="impactt", beam=_beam(lat), workdir=tmp_path / "it")
    k = next(i for i, n in enumerate(r.names) if n.startswith("cavity"))
    gain = r.ref_kinetic_eV_out[k] - r.ref_kinetic_eV_in[k]
    assert gain == pytest.approx(V_VOLT * math.cos(math.radians(30.0)), rel=2e-5)   # MEASURED 3.3e-6
    assert r.to_common().R_elem[k][5, 4] < 0
