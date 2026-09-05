"""Cheetah as an engine (Phase 5.3): the FODO gate against cpymad, TraceWin decks against HELIX, the
negative-charge conventions, and the audit of the zero-length cavity.  Runs where Cheetah is installed
(conda env ``cheetah`` or ``LATTIX_CHEETAH_PYTHON``; marker ``oracle_cheetah``)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import _beam
from lattix.oracles import get_oracle
from lattix.oracles.base import BeamSpec
from lattix.oracles.compare import compare_pair
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, V_VOLT, write_decks

pytestmark = pytest.mark.oracle_cheetah
DATA = Path(__file__).resolve().parents[1] / "data" / "public"


@pytest.fixture(scope="module")
def cheetah():
    o = get_oracle("cheetah")
    ok, why = o.available()
    if not ok:
        pytest.skip(f"Cheetah unavailable: {why}")
    return o


@pytest.mark.oracle_madx
def test_fodo_matches_cpymad_on_every_block(cheetah, tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.cheetah.json"
    rep = write(lat, out, "cheetah")
    assert rep.ok
    rc = cheetah.run(out, fmt="cheetah", beam=_beam(lat), workdir=tmp_path / "ch")
    rm = get_oracle("madx").run(DATA / "helix" / "fodo.madx", fmt="madx", beam=_beam(lat), workdir=tmp_path / "madx")
    pc = compare_pair(rm, rc)
    assert pc.n_shared == 8
    assert pc.blocks["T4x4"] < 1e-8 and pc.blocks["disp"] < 1e-8 and pc.blocks["path"] < 1e-8
    assert pc.blocks["R56"] < 1e-7


@pytest.mark.oracle_helix
@pytest.mark.parametrize("deck, sp, tol", [("fodo_cell.dat", "proton", 1e-8), ("solenoid_channel.dat", "proton", 1e-8),
                                            ("mebt_line.dat", "h-", 2e-2)])
def test_tracewin_decks_against_helix(cheetah, tmp_path, deck, sp, tol):
    """Quads and solenoids exact (the solenoid k = B/(2Bρ_signed) and k1 = G/Bρ_signed conventions);
    the MEBT's two thin gaps carry lattix's thin-gap RF-focusing lens next to a zero-length Cavity the
    oracle tracks as 1 µm: Equivalent tier (measured 6.5e-3)."""
    helix = get_oracle("helix")
    if not helix.available()[0]:
        pytest.skip("HELIX not available")
    lat, _ = read(DATA / "helix" / deck, "tracewin", species=sp, kinetic_energy_eV=2.1e6, frequency_Hz=162.5e6)
    out = tmp_path / (Path(deck).stem + ".cheetah.json")
    rep = write(lat, out, "cheetah")
    rc = cheetah.run(out, fmt="cheetah", beam=_beam(lat), workdir=tmp_path / "ch")
    rh = helix.run(DATA / "helix" / deck, fmt="tracewin", beam=_beam(lat), workdir=tmp_path / "hx")
    pc = compare_pair(rh, rc)
    assert pc.n_shared >= 8
    assert pc.blocks["T4x4"] < tol and pc.blocks["disp"] < tol
    assert pc.energy_rel < 1e-9
    if deck == "mebt_line.dat":
        assert rep.codes()["CHEETAH_ZERO_LENGTH_CAVITY"] == 2
        assert any("zero-length Cavity" in w for w in rc.warnings)


def test_negative_charge_accelerates_at_the_same_phase(cheetah, tmp_path):
    """Cheetah's gain is −voltage·q·cos(phase): the writer's voltage = −V/q keeps the IR's
    species-independent V·cos φ for protons and H⁻ alike (fingerprint deck, φs = −30°)."""
    deck, fmt = write_decks("cheetah", tmp_path)["cavity"]
    r = cheetah.run(deck, fmt=fmt, beam=BeamSpec(species="proton", kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ),
                    workdir=tmp_path / "p")
    j = r.names.index("c")
    gain = V_VOLT * math.cos(math.radians(30.0))
    assert r.ref_kinetic_eV_out[j] - r.ref_kinetic_eV_in[j] == pytest.approx(gain, rel=1e-9)
    from lattix.ir.elements import RFP, Drift, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("h-"), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", length=0.02,
                                               rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0),
                                                      frequency_Hz=FREQ_HZ)),
                                      Drift(name="d2", length=0.5)], ref)
    out = tmp_path / "hminus.cheetah.json"
    write(lat, out, "cheetah")
    rh = cheetah.run(out, fmt="cheetah", beam=_beam(lat), workdir=tmp_path / "h")
    k = rh.names.index("c")
    assert rh.ref_kinetic_eV_out[k] - rh.ref_kinetic_eV_in[k] == pytest.approx(gain, rel=1e-9)
    assert rh.charge == -1


@pytest.mark.oracle_madx
def test_cheetahs_own_bmad_converter_agrees_with_the_lattix_cheetah_writer(cheetah, tmp_path):
    """Lockstep through a third party: the ``.bmad`` lattix writes for ``fodo.madx``, read by Cheetah's
    own ``converters.bmad``, gives the same Cheetah maps as lattix's LatticeJSON of the same lattice."""
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    via_json = tmp_path / "fodo.cheetah.json"
    via_bmad = tmp_path / "fodo.bmad"
    write(lat, via_json, "cheetah")
    write(lat, via_bmad, "bmad")
    rj = cheetah.run(via_json, fmt="cheetah", beam=_beam(lat), workdir=tmp_path / "j")
    rb = cheetah.run(via_bmad, fmt="bmad", beam=_beam(lat), workdir=tmp_path / "b")
    pc = compare_pair(rj, rb)
    assert pc.n_shared >= 6
    assert pc.blocks["T4x4"] < 1e-8 and pc.blocks["disp"] < 1e-8
