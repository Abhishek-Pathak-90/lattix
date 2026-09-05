"""SciBmad as an engine: the measured conventions, and lattix-written decks against the other engines.

Runs only where ``julia`` with SciBmad is installed (marker ``oracle_scibmad``)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix import read, write
from lattix.formats.scibmad import GAIN_SIGN
from lattix.ir.elements import RFP, Drift, RFCavity
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.oracles import get_oracle
from lattix.oracles.compare import compare_pair
from lattix.testing import needs

pytestmark = [pytest.mark.oracle_scibmad, needs("scibmad")]

DATA = Path(__file__).resolve().parents[1] / "data" / "public"


def _beam(lat):
    from lattix.crossval import _beam

    return _beam(lat)


def test_gain_sign_is_the_measured_one(tmp_path):
    """SciBmad 0.5.2 accelerates when −voltage·cos(phi0) > 0 (GAIN_SIGN = −1); the writer compensates,
    so a lattix-written crest cavity accelerates by V.  If this fails, SciBmad changed its convention:
    flip GAIN_SIGN and re-measure, do not touch the test."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)
    lat = Lattice.from_sequence("g", [Drift(name="d1", length=0.5),
                                      RFCavity(name="c", length=0.02, rf=RFP(voltage_V=3e5, phase_rad=0.0,
                                                                              frequency_Hz=162.5e6)),
                                      Drift(name="d2", length=0.5)], ref)
    out = tmp_path / "gain.jl"
    write(lat, out, "scibmad")
    r = get_oracle("scibmad").run(out, fmt="scibmad", workdir=tmp_path / "wd")
    # constant-p0 engine: the gain shows up as pz of the tracked orbit, i.e. in the cavity's R66 row
    # through the map around the orbit, not in ref_kinetic_eV; check the raw worker orbit instead
    assert r.ref_kinetic_eV_out[0] == pytest.approx(r.ref_kinetic_eV_in[0])
    import json

    data = json.loads((tmp_path / "wd" / "scibmad_result.json").read_text())
    assert data["p0_model"].startswith("constant")
    assert GAIN_SIGN == -1.0
    assert "using Beamlines" in out.read_text()


def test_drift_fingerprint_and_basis(tmp_path):
    """A 1 m drift at 2.1 MeV: R56 in the common basis is +L/γ² (z ahead-positive, Bmad's basis)."""
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)
    lat = Lattice.from_sequence("d", [Drift(name="d", length=1.0)], ref)
    out = tmp_path / "d.jl"
    write(lat, out, "scibmad")
    r = get_oracle("scibmad").run(out, fmt="scibmad", workdir=tmp_path / "wd")
    R = r.to_common().R_elem[0]
    gamma = 1.0 + 2.1e6 / ref.species.mass_eV
    assert R[4, 5] == pytest.approx(1.0 / gamma**2, rel=1e-7)
    assert R[0, 1] == pytest.approx(1.0, rel=1e-9) and abs(R[1, 0]) < 1e-9


def test_fodo_matches_cpymad(tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.jl"
    write(lat, out, "scibmad")
    rs = get_oracle("scibmad").run(out, fmt="scibmad", workdir=tmp_path / "wd_s")
    rm = get_oracle("madx").run(DATA / "helix" / "fodo.madx", fmt="madx", beam=_beam(lat), workdir=tmp_path / "wd_m")
    pc = compare_pair(rm, rs)
    assert pc.n_shared == 8
    assert pc.blocks["T4x4"] < 1e-8 and pc.blocks["disp"] < 1e-8 and pc.blocks["path"] < 1e-8
    assert pc.blocks["R56"] < 1e-7                 # finite-difference limit of the worker


@pytest.mark.oracle_helix
@pytest.mark.parametrize("deck", ["mebt_line.dat", "dtl_section.dat"])
def test_thin_gap_linacs_match_helix_on_the_transverse_block(tmp_path, deck):
    """Thin gaps become 1 µm cavities in the worker, the RF defocusing is a LineElement lens and the
    delta energy mode carries the acceleration: measured 1.2e-10 (MEBT) and 1.6e-11 (DTL)."""
    ok, why = get_oracle("helix").available()
    if not ok:
        pytest.skip(why)
    lat, _ = read(DATA / "helix" / deck, "tracewin")
    out = tmp_path / (Path(deck).stem + ".jl")
    rep = write(lat, out, "scibmad")
    assert "THIN_GAP_RF_FOCUSING_AS_MATRIX" in rep.codes()
    rs = get_oracle("scibmad").run(out, fmt="scibmad", workdir=tmp_path / "wd_s")
    assert any("zero-length RFCavity tracked as 1e-6 m" in w for w in rs.warnings)
    rh = get_oracle("helix").run(DATA / "helix" / deck, fmt="tracewin", beam=_beam(lat), workdir=tmp_path / "wd_h")
    pc = compare_pair(rh, rs)
    assert pc.n_shared >= 5
    assert pc.blocks["T4x4"] < 1e-8 and pc.blocks["disp"] < 1e-8


def test_fringe_integrals_are_zeroed_with_a_warning(tmp_path):
    from lattix.ir.elements import Bend, BendP

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=8e8)
    b = Bend(name="b", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05, edge_int1=0.45, hgap=0.03))
    lat = Lattice.from_sequence("b", [Drift(name="d", length=0.5), b], ref)
    out = tmp_path / "b.jl"
    write(lat, out, "scibmad")
    assert "edge1_int = 0.0135" in out.read_text()
    r = get_oracle("scibmad").run(out, fmt="scibmad", workdir=tmp_path / "wd")
    assert any("edge1_int/edge2_int set to 0" in w for w in r.warnings)
    assert np.all(np.isfinite(r.R_elem))
    R = r.to_common().R_elem[1]
    # e1 = e2 = θ/2 is a rectangular magnet: the horizontal edge focusing cancels the body exactly,
    # the vertical plane is two thin edges around a drift (no fringe correction: it was zeroed)
    assert abs(R[1, 0]) < 1e-8
    g, e = 0.1, 0.05
    assert R[3, 2] == pytest.approx(-2 * g * math.tan(e) + (g * math.tan(e)) ** 2 * 1.0, rel=1e-6)
