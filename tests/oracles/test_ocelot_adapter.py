"""Ocelot as an engine (Phase 5.7): the FODO gate against cpymad with an electron, the fingerprint
cavity's gain and slope, a proton deck as report-only, the executed-module reader fallback and the
XFEL ``.lte`` lockstep through Ocelot's own Elegant converter.  Runs where Ocelot is installed (env
``ocelot`` or ``LATTIX_OCELOT_PYTHON``; marker ``oracle_ocelot``)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix import read, write
from lattix.crossval import _beam
from lattix.ir.reference import ReferenceParticle, species
from lattix.oracles import get_oracle
from lattix.oracles.base import BeamSpec
from lattix.oracles.compare import compare_pair
from lattix.oracles.fingerprint import FREQ_HZ, KE_EV, V_VOLT, write_decks
from lattix.testing import codes_path

pytestmark = pytest.mark.oracle_ocelot
DATA = Path(__file__).resolve().parents[1] / "data" / "public"
OCELOT_CLONE = codes_path('tier2_modern', 'ocelot')
XFEL_LTE = OCELOT_CLONE / "unit_tests/adaptors_test/elegant_lattice/ref_results/XFEL_elegant_TD1_S2E.lte"


@pytest.fixture(scope="module")
def ocelot():
    o = get_oracle("ocelot")
    ok, why = o.available()
    if not ok:
        pytest.skip(f"Ocelot unavailable: {why}")
    return o


def _electron_fodo():
    """HELIX's fodo.madx (quads + two sector bends) with a 1 GeV electron as the reference: the same
    fields, an electron's normalized strengths in both decks."""
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    lat.reference = ReferenceParticle(species=species("electron"), kinetic_energy_eV=1e9)
    return lat


@pytest.mark.oracle_madx
def test_fodo_matches_cpymad_on_every_block(ocelot, tmp_path):
    lat = _electron_fodo()
    out = tmp_path / "fodo.ocelot.py"
    madx = tmp_path / "fodo.madx"
    rep = write(lat, out, "ocelot")
    assert rep.ok
    write(lat, madx, "madx")
    ro = ocelot.run(out, fmt="ocelot", beam=_beam(lat), workdir=tmp_path / "oc")
    rm = get_oracle("madx").run(madx, fmt="madx", beam=_beam(lat), workdir=tmp_path / "madx")
    pc = compare_pair(rm, ro)
    assert pc.n_shared == 8
    assert pc.blocks["T4x4"] < 1e-8 and pc.blocks["disp"] < 1e-8 and pc.blocks["path"] < 1e-8
    assert pc.blocks["R56"] < 1e-7
    assert ro.meta["electron_only"] is False and not ro.warnings


def test_fingerprint_cavity_gains_and_bunches(ocelot, tmp_path):
    """The surrogate cavity for a 1 MV thin gap at φs = −30°: the reference gains V·cos 30° exactly
    (Ocelot's ``v·cos(phi)``) and a late particle gains more (R65 < 0 in the common basis)."""
    deck, fmt = write_decks("ocelot", tmp_path)["cavity"]
    r = ocelot.run(deck, fmt=fmt, beam=BeamSpec(species="electron", kinetic_energy_eV=KE_EV, frequency_Hz=FREQ_HZ),
                   workdir=tmp_path / "e")
    j = r.names.index("c")
    assert r.ref_kinetic_eV_out[j] - r.ref_kinetic_eV_in[j] == pytest.approx(V_VOLT * math.cos(math.radians(30.0)),
                                                                             rel=1e-12)
    assert r.to_common().R_elem[j][5, 4] < 0 and r.R_elem[j][5, 4] > 0
    assert r.total_length == pytest.approx(1.0)


def test_thin_gap_surrogate_against_a_thick_cavity_of_the_same_voltage(ocelot, tmp_path):
    """MEASURED (docs/oracles.md, Phase 5.7): Ocelot's cavity (Rosenzweig–Serafini edges + body) has a
    transverse focusing that scales with V/(E·l) — a 1 MV gap at 2.1 MeV gives R21 = −1.48 over 2 cm and
    −0.79 over the 37 mm surrogate — so a thin gap has no length-free limit in Ocelot: the surrogate
    length is a recorded modelling choice (``THIN_GAP_AS_SHORT_CAVITY``, ``thin_gap_length_m=`` to
    override); the energy gain does not depend on it."""
    from lattix.ir.elements import RFP, Drift, RFCavity
    from lattix.ir.lattice import Lattice

    ref = ReferenceParticle(species=species("electron"), kinetic_energy_eV=KE_EV, rf_frequency_Hz=FREQ_HZ)
    els = [Drift(name="d1", length=0.5),
           RFCavity(name="c", length=0.02, rf=RFP(voltage_V=V_VOLT, phase_rad=math.radians(-30.0),
                                                  frequency_Hz=FREQ_HZ)),
           Drift(name="d2", length=0.5)]
    lat = Lattice.from_sequence("t", els, ref)
    thick = tmp_path / "thick.ocelot.py"
    write(lat, thick, "ocelot")
    rt = ocelot.run(thick, fmt="ocelot", beam=_beam(lat), workdir=tmp_path / "t")
    deck, fmt = write_decks("ocelot", tmp_path)["cavity"]
    rs = ocelot.run(deck, fmt=fmt, beam=_beam(lat), workdir=tmp_path / "s")
    jt, js = rt.names.index("c"), rs.names.index("c")
    assert rt.ref_kinetic_eV_out[jt] - rt.ref_kinetic_eV_in[jt] == pytest.approx(
        rs.ref_kinetic_eV_out[js] - rs.ref_kinetic_eV_in[js], rel=1e-12)
    r21_thick, r21_surrogate = rt.R_elem[jt][1, 0], rs.R_elem[js][1, 0]
    assert r21_thick < r21_surrogate < 0                    # both focus, the shorter one more strongly
    assert abs(r21_thick * rt.length[jt] - r21_surrogate * rs.length[js]) / abs(r21_thick * rt.length[jt]) < 0.2


def test_proton_deck_is_report_only(ocelot, tmp_path):
    lat, _ = read(DATA / "helix" / "fodo_cell.dat", "tracewin", species="proton", kinetic_energy_eV=2.1e6,
                  frequency_Hz=162.5e6)
    out = tmp_path / "fodo_cell.ocelot.py"
    rep = write(lat, out, "ocelot")
    assert "OCELOT_ELECTRON_ONLY" in rep.codes()
    r = ocelot.run(out, fmt="ocelot", beam=_beam(lat), workdir=tmp_path / "p")
    assert r.meta["electron_only"] is True and any("electron" in w for w in r.warnings)
    helix = get_oracle("helix")
    if helix.available()[0]:
        # the same γ is handed to Ocelot, so quads and drifts (no m_e in their maps) still agree
        rh = helix.run(DATA / "helix" / "fodo_cell.dat", fmt="tracewin", beam=_beam(lat), workdir=tmp_path / "h")
        pc = compare_pair(rh, r)
        assert pc.blocks["T4x4"] < 1e-8


def test_reader_fallback_runs_the_module_in_ocelot(ocelot, tmp_path):
    p = tmp_path / "loop.py"
    p.write_text("from ocelot import *\ncell = []\nfor i in range(3):\n    cell.append(Quadrupole(l=0.2, k1=(-1)**i, "
                 "eid='q%d' % i))\n    cell.append(Drift(l=1.0, eid='d%d' % i))\nlat = MagneticLattice(cell)\n")
    lat, rep = read(p, species="electron", kinetic_energy_eV=1e9, use_ocelot=True)
    assert "OCELOT_EXECUTED" in rep.codes()
    names = [it.ref for it in lat.lines[lat.use].items]
    assert names == ["q0", "d0", "q1", "d1", "q2", "d2"] and lat.total_length == pytest.approx(3.6)
    assert lat.elements["q1"].multipole.Bn[1] == pytest.approx(-lat.reference.brho_signed)


@pytest.mark.skipif(not XFEL_LTE.is_file(), reason="Ocelot clone with the XFEL .lte not present")
def test_xfel_lockstep_through_ocelots_own_elegant_converter(ocelot, tmp_path):
    """The XFEL S2E Elegant deck read by lattix (Elegant reader → Ocelot writer) against Ocelot's own
    ``ElegantLatticeConverter``: two paths to Ocelot maps that share nothing."""
    lat, rep = read(XFEL_LTE, "elegant", species="electron", kinetic_energy_eV=130e6)
    out = tmp_path / "xfel.ocelot.py"
    write(lat, out, "ocelot")
    beam = _beam(lat)
    ra = ocelot.run(out, fmt="ocelot", beam=beam, workdir=tmp_path / "lattix")
    rb = ocelot.run(XFEL_LTE, fmt="elegant", beam=beam, workdir=tmp_path / "ocelot")
    pc = compare_pair(ra, rb)
    assert pc.n_shared > 100
    assert pc.blocks["T4x4"] < 1e-6
