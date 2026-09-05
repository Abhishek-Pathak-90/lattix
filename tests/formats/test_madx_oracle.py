"""Engine-backed checks for the MAD-X reader/writer pair (PLAN §5.6 A1 / A3 MAD-X leg).

Nothing here is round-trip-only: every assertion runs real MAD-X (cpymad) on
*both* the original deck and the deck lattix wrote, and compares the per-element
optics through :func:`lattix.oracles.compare.compare_pair`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix.formats.madx import Reader, Writer
from lattix.ir.elements import (
    RFP,
    Drift,
    MagneticMultipoleP,
    Marker,
    Quadrupole,
    ReferenceChange,
    RFCavity,
)
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.oracles import Probe, get_oracle
from lattix.oracles.compare import compare_pair
from lattix.testing import needs, require

require("madx")
pytestmark = pytest.mark.oracle_madx

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
TOL = 1e-9


@needs("madx")
@pytest.mark.parametrize("deck", ["fodo.madx", "transport.madx"])
def test_written_deck_matches_the_source_in_madx(deck, tmp_path):
    src = DATA / deck
    lat, rep_in = Reader().read(src)
    out = tmp_path / f"lattix_{deck}"
    rep_out = Writer().write(lat, out, strict=True)
    assert rep_in.ok and rep_out.ok

    oracle = get_oracle("madx")
    a = oracle.run(src, workdir=tmp_path / "a")
    b = oracle.run(out, workdir=tmp_path / "b")
    cmp = compare_pair(a, b)

    assert cmp.n_shared >= 8
    assert cmp.length_a == cmp.length_b == pytest.approx(lat.total_length, abs=1e-12)
    assert cmp.blocks["T4x4"] < TOL
    assert cmp.blocks["disp"] < TOL
    assert cmp.blocks["path"] < TOL
    assert cmp.max_rcum_abs < TOL
    assert cmp.notes == []


@needs("madx")
def test_written_fodo_reproduces_the_golden_twiss(tmp_path):
    """golden.yaml pins βx/βy at the end of fodo.madx; the written deck must agree."""
    lat, _ = Reader().read(DATA / "fodo.madx")
    out = tmp_path / "fodo.madx"
    Writer().write(lat, out)
    oracle = get_oracle("madx")
    a = oracle.run(DATA / "fodo.madx", workdir=tmp_path / "a")
    b = oracle.run(out, workdir=tmp_path / "b")
    for key in ("betx", "bety", "alfx", "alfy"):
        assert b.twiss[key][-1] == pytest.approx(a.twiss[key][-1], rel=1e-12), key
    assert b.survey[-1].tolist() == pytest.approx(a.survey[-1].tolist(), abs=1e-12)


@needs("madx")
def test_thin_cavity_reference_gain_is_v_cos_phi(tmp_path):
    """I-6 / the phase rule: MAD-X's own tracking must give ΔE = V·cos(φs)."""
    volt, phase, ke = 1e6, -math.pi / 6, 2.1e6
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke)
    lat = Lattice.from_sequence("cav", [
        Drift(name="d0", length=0.5),
        RFCavity(name="gap", length=0.0,
                 rf=RFP(voltage_V=volt, phase_rad=phase, frequency_Hz=162.5e6)),
        Drift(name="d1", length=0.5),
        Marker(name="end"),
    ], ref)
    out = tmp_path / "cav.madx"
    rep = Writer().write(lat, out)
    assert "CONST_P0" in rep.codes()                 # MAD-X cannot follow p0
    assert f"lag={(phase / (2 * math.pi) + 0.25):.15g}" in out.read_text()

    res = get_oracle("madx").run(out, probe=Probe(coords=np.zeros((1, 6))),
                                 workdir=tmp_path / "run")
    m = res.mass_eV
    pc = math.sqrt((ke + m) ** 2 - m * m)
    dE = float(res.probe_out[0, 5]) * pc            # MAD-X pt = ΔE / (p0·c)
    assert dE == pytest.approx(volt * math.cos(phase), rel=1e-9)
    assert dE == pytest.approx(866_025.4037844387, rel=1e-9)


@needs("madx")
def test_hminus_deck_flips_the_gradient_and_stays_optically_identical(tmp_path):
    """I-8: an H⁻ deck's k1 keeps its sign through IR → MAD-X (signed rigidity)."""
    src = tmp_path / "h.madx"
    m_h = (938_272_088.16 + 2 * 510_998.95) / 1e9
    src.write_text(f"""
beam, particle=ion, mass={m_h!r}, charge=-1, energy={m_h + 0.8!r};
qf: quadrupole, l=0.3, k1=0.6;
qd: quadrupole, l=0.3, k1=-0.6;
s: sequence, l=3.0, refer=centre;
  qf, at=0.5;
  qd, at=2.0;
endsequence;
use, sequence=s;
""")
    lat, _ = Reader().read(src)
    assert lat.elements["qf"].gradient < 0          # H⁻: signed Bρ < 0
    out = tmp_path / "written.madx"
    Writer().write(lat, out, strict=True)
    oracle = get_oracle("madx")
    cmp = compare_pair(oracle.run(src, workdir=tmp_path / "a"),
                       oracle.run(out, workdir=tmp_path / "b"))
    assert cmp.blocks["T4x4"] < TOL
    assert cmp.max_rcum_abs < TOL


@needs("madx")
def test_local_energy_mode_gives_the_right_optics_per_section(tmp_path):
    """The rule under test: downstream of an energy change, k1 must use the *local* Bρ.

    A :class:`ReferenceChange` moves the IR reference energy without putting an
    accelerating element in the MAD-X deck (it is written as a marker), so MAD-X's
    orbit stays at pt = 0 and the two writer modes differ only in the k1 they print.
    The independent reference is a separately written deck whose BEAM already sits
    at the post-change energy.
    """
    ke0, dE, grad = 1e8, 5e7, 2.0                    # eV, eV, T/m (one physical magnet)
    ref0 = ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke0)
    ref1 = ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke0 + dE)
    b0, b1 = ref0.brho_signed, ref1.brho_signed
    assert b1 / b0 == pytest.approx(1.240139111470995, rel=1e-12)

    lat = Lattice.from_sequence("lin", [
        Quadrupole(name="qa", length=0.3, multipole=MagneticMultipoleP(Bn={1: grad})),
        Drift(name="d1", length=0.5),
        ReferenceChange(name="rc", dE_ref_eV=dE),
        # no drift after it, so MAD-X's sectortable row for qb is the quadrupole alone
        Quadrupole(name="qb", length=0.3, multipole=MagneticMultipoleP(Bn={1: grad})),
        Marker(name="end"),
    ], ref0)
    loc, con = tmp_path / "loc.madx", tmp_path / "con.madx"
    rep_loc = Writer().write(lat, loc, energy_mode="local")
    rep_con = Writer().write(lat, con, energy_mode="constant")
    assert "CONST_P0_LOCAL_RIGIDITY" in rep_loc.codes()
    assert "CONST_P0_START_RIGIDITY" in rep_con.codes()
    assert "REFCHANGE_AS_TAG" in rep_loc.codes()

    check = tmp_path / "check.madx"
    check.write_text(f"""
beam, particle=proton, mass={ref1.species.mass_eV / 1e9!r}, charge=1,
      energy={ref1.total_energy_eV / 1e9!r};
qb: quadrupole, l=0.3, k1={grad / b1!r};
s: sequence, l=0.3, refer=centre; qb, at=0.15; endsequence;
use, sequence=s;
""")

    oracle = get_oracle("madx")
    r = {}
    for name, deck in (("loc", loc), ("con", con), ("check", check)):
        res = oracle.run(deck, workdir=tmp_path / name)
        r[name] = res.R_elem[res.names.index("qb")]

    assert np.max(np.abs(r["loc"][:4, :4] - r["check"][:4, :4])) < 1e-12
    assert np.max(np.abs(r["con"][:4, :4] - r["check"][:4, :4])) > 1e-3
    assert abs(r["loc"][1, 0]) < abs(r["con"][1, 0])          # weaker at higher energy


@needs("madx")
def test_madx_twiss_orbit_reacts_to_an_accelerating_cavity(tmp_path):
    """Why an accelerating deck is EQUIVALENT and never EXACT (measured 2026-09-03).

    MAD-X keeps p0 constant but its TWISS *does* carry the cavity's energy gain in
    the reference orbit's ``pt`` and linearises every downstream magnet about it —
    so the map of a magnet after a real cavity is not the map it would have in a
    deck without one.  ``EQUIVALENT:CONST_P0`` on the cavity is the ledger entry
    that says the reference model differs; the number below pins the effect.
    """
    from cpymad.madx import Madx

    ke, dE = 1e8, 5e7
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=ke)
    lat = Lattice.from_sequence("lin", [
        Quadrupole(name="qa", length=0.3, multipole=MagneticMultipoleP(Bn={1: 2.0})),
        Drift(name="d1", length=0.5),
        RFCavity(name="cav", length=0.0,
                 rf=RFP(voltage_V=dE, phase_rad=0.0, frequency_Hz=325e6)),
        Quadrupole(name="qb", length=0.3, multipole=MagneticMultipoleP(Bn={1: 2.0})),
        Marker(name="end"),
    ], ref)
    deck = tmp_path / "acc.madx"
    rep = Writer().write(lat, deck, energy_mode="constant")
    assert rep.codes()["CONST_P0"] == 1
    assert "volt=50," in deck.read_text()             # V in MV

    madx = Madx(stdout=False)
    madx.chdir(str(tmp_path))
    madx.call(str(deck))
    tw = madx.twiss(sequence="lin", betx=10, bety=10)
    pt_end = float(np.asarray(tw.pt, dtype=float)[-1])
    madx.quit()
    # pt = ΔE/(p0·c): the reference orbit really does pick up the gain
    pc0 = math.sqrt((ke + ref.species.mass_eV) ** 2 - ref.species.mass_eV ** 2)
    assert pt_end == pytest.approx(dE / pc0, rel=1e-6)
    assert pt_end == pytest.approx(0.11246483272571714, rel=1e-9)
