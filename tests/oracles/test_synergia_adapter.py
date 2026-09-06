"""Synergia 3 as an engine (Phase 5.9, local build): the FODO gate against cpymad, the measured constant-p0
semantics (strengths scaled by p_design/p_bunch, no phase slip, under-bent bends), the solenoid bug of the
clone, and a native archive through Synergia's own loader.  Runs where a built Synergia is at hand
(``LATTIX_SYNERGIA_PYTHON`` / the clone's pixi install; marker ``oracle_synergia``)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix import read, write
from lattix.crossval import _beam
from lattix.ir.elements import RFP, Bend, BendP, Drift, MagneticMultipoleP, Quadrupole, RFCavity, Solenoid, SolenoidP
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.walk import propagate
from lattix.oracles import get_oracle
from lattix.oracles.base import BeamSpec
from lattix.oracles.compare import compare_pair
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, V_VOLT, write_decks

pytestmark = pytest.mark.oracle_synergia
DATA = Path(__file__).resolve().parents[1] / "data" / "public"


@pytest.fixture(scope="module")
def synergia():
    o = get_oracle("synergia")
    ok, why = o.available()
    if not ok:
        pytest.skip(f"Synergia unavailable: {why}")
    return o


def _ref(sp="proton", ke=KE_EV):
    return ReferenceParticle(species=species(sp), kinetic_energy_eV=ke, rf_frequency_Hz=FREQ_HZ)


@pytest.mark.oracle_madx
def test_fodo_matches_cpymad_on_every_block(synergia, tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.synergia.json"
    rep = write(lat, out, "synergia")
    assert rep.ok
    rs = synergia.run(out, fmt="synergia", beam=_beam(lat), workdir=tmp_path / "syn")
    rm = get_oracle("madx").run(DATA / "helix" / "fodo.madx", fmt="madx", beam=_beam(lat), workdir=tmp_path / "madx")
    pc = compare_pair(rm, rs)
    assert pc.n_shared == 8
    # MEASURED 4.5e-8: Synergia's quadrupole is a second-order Yoshida integrator, not the exact matrix
    assert pc.blocks["T4x4"] < 1e-7 and pc.blocks["disp"] < 1e-7 and pc.blocks["path"] < 1e-7
    assert pc.blocks["R56"] < 1e-7


def test_fingerprint_cavity_gains_and_bunches(synergia, tmp_path):
    """lag = φ/2π + ¼: the bunch reference gains V·cos 30° to machine precision and a late particle gains more."""
    deck, fmt = write_decks("synergia", tmp_path)["cavity"]
    r = synergia.run(deck, fmt=fmt, beam=BeamSpec(species="proton", kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ),
                     workdir=tmp_path / "c")
    j = r.names.index("c")
    assert r.ref_kinetic_eV_out[j] - r.ref_kinetic_eV_in[j] == pytest.approx(V_VOLT * math.cos(math.radians(30.0)),
                                                                             rel=1e-12)
    assert r.to_common().R_elem[j][5, 4] < 0
    ref = _ref("h-")
    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0),
                                                                frequency_Hz=FREQ_HZ)),
                                      Drift(name="d2", length=0.5)], ref)
    out = tmp_path / "hminus.synergia.json"
    write(lat, out, "synergia")
    rh = synergia.run(out, fmt="synergia", beam=_beam(lat), workdir=tmp_path / "h")
    k = rh.names.index("c")
    assert rh.ref_kinetic_eV_out[k] - rh.ref_kinetic_eV_in[k] == pytest.approx(V_VOLT * math.cos(math.radians(30.0)),
                                                                               rel=1e-12)
    assert rh.charge == -1 and rh.to_common().R_elem[k][5, 4] < 0


def test_constant_design_momentum_semantics(synergia, tmp_path):
    """MEASURED: two 1 MV gaps at −30° then a quadrupole — the second gap gains V·cos 30° with the plain lag
    (no phase slip) and the quadrupole written with the start rigidity (k1 = G/Bρ_start) shows the lab
    gradient at the local momentum (Synergia scales strengths by p_design/p_bunch); the "local" policy
    would double-correct it."""
    ref = _ref()
    els = [Drift(name="d0", length=0.5),
           RFCavity(name="c1", rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0), frequency_Hz=FREQ_HZ)),
           Drift(name="d1", length=0.5),
           RFCavity(name="c2", rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0), frequency_Hz=FREQ_HZ)),
           Drift(name="d2", length=0.5),
           Quadrupole(name="q", length=0.2, multipole=MagneticMultipoleP(Bn={1: 1.2}))]
    lat = Lattice.from_sequence("two", els, ref)
    q = next(p for p in propagate(lat) if p.element.name == "q")
    k_local = 1.2 / q.ref_in.brho_signed

    def r21(k):
        return -math.sqrt(k) * math.sin(math.sqrt(k) * 0.2)

    for mode, expect in (("constant", r21(k_local)), ("local", None)):
        out = tmp_path / f"two_{mode}.synergia.json"
        write(lat, out, "synergia", energy_mode=mode)
        r = synergia.run(out, fmt="synergia", beam=_beam(lat), workdir=tmp_path / mode)
        gains = {n: r.ref_kinetic_eV_out[i] - r.ref_kinetic_eV_in[i] for i, n in enumerate(r.names)}
        assert gains["c1"] == pytest.approx(V_VOLT * math.cos(math.radians(30.0)), rel=1e-12)
        assert gains["c2"] == pytest.approx(V_VOLT * math.cos(math.radians(30.0)), rel=1e-12)
        R21 = r.R_elem[r.names.index("q")][1, 0]
        if expect is not None:
            assert R21 == pytest.approx(expect, rel=1e-6)                   # Yoshida integrator: 8e-8
        else:                                          # "local": k1 = G/Bρ_local, scaled down once more by Synergia
            assert R21 == pytest.approx(r21(k_local * ref.brho_signed / q.ref_in.brho_signed), rel=1e-6)


def test_bend_after_acceleration_is_underbent(synergia, tmp_path):
    """MEASURED: a 10° sector bend after a crest 1 MV gap at 2.1 MeV bends the bunch by angle·p_design/p_local
    (R21 = −h·sin(θ_eff)); the writer records LOSSY CONST_P0_BEND_UNDERBENT."""
    ref = _ref()
    lat = Lattice.from_sequence("gb", [Drift(name="d0", length=0.5),
                                       RFCavity(name="c1", rf=RFP(voltage_V=V_VOLT, phase_rad=0.0,
                                                                  frequency_Hz=FREQ_HZ)),
                                       Drift(name="d1", length=0.5), Bend(name="b", length=1.0, bend=BendP(angle=0.1)),
                                       Drift(name="d2", length=0.5)], ref)
    out = tmp_path / "gb.synergia.json"
    rep = write(lat, out, "synergia")
    assert "CONST_P0_BEND_UNDERBENT" in rep.codes()
    r = synergia.run(out, fmt="synergia", beam=_beam(lat), workdir=tmp_path / "gb")
    b = next(p for p in propagate(lat) if p.element.name == "b")
    theta_eff = 0.1 * ref.pc_eV / b.ref_in.pc_eV
    assert r.R_elem[r.names.index("b")][1, 0] == pytest.approx(-0.1 * math.sin(theta_eff), rel=1e-3)


def test_solenoid_body_bug_of_the_clone(synergia, tmp_path):
    """MEASURED (clone 17e691d): ff_solenoid calls solenoid_unit(…, ksl, ks, …) whose signature is (ks, ksl):
    the body rotates by ks and divides the displacement by ks·L — R12 = sin(ks)/(ks·L) instead of a drift-like
    length.  The writer keeps MAD-X's ks = B/Bρ; solenoid decks are report-only in the battery."""
    ref = _ref()
    ks = 0.1
    lat = Lattice.from_sequence("s", [Drift(name="d0", length=0.1),
                                      Solenoid(name="s", length=0.3, solenoid=SolenoidP(Bsol_T=ks * ref.brho_signed)),
                                      Drift(name="d1", length=0.1)], ref)
    out = tmp_path / "sol.synergia.json"
    write(lat, out, "synergia")
    r = synergia.run(out, fmt="synergia", beam=_beam(lat), workdir=tmp_path / "s")
    R = r.R_elem[r.names.index("s")]
    assert R[0, 1] == pytest.approx(math.sin(ks) / (ks * 0.3), rel=1e-6)          # the swapped arguments
    assert abs(np.linalg.det(R[:4, :4]) - 1.0) < 1e-9


def test_native_archive_round_trip_through_synergia(synergia, tmp_path):
    """A MAD-X FODO written by lattix, loaded by Synergia and re-read: the same maps."""
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.synergia.json"
    write(lat, out, "synergia")
    back, _ = read(out)
    out2 = tmp_path / "fodo2.synergia.json"
    write(back, out2, "synergia")
    ra = synergia.run(out, fmt="synergia", beam=_beam(lat), workdir=tmp_path / "a")
    rb = synergia.run(out2, fmt="synergia", beam=_beam(lat), workdir=tmp_path / "b")
    assert np.max(np.abs(ra.R_cum[-1] - rb.R_cum[-1])) < 1e-12
