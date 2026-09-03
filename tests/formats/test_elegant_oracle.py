"""Engine-backed checks for the Elegant reader/writer pair (PLAN §5.6 A2, elegant leg).

Nothing here is round-trip-only: every assertion runs the real ``elegant`` binary
on the deck lattix wrote, and — where a second engine can see the same machine —
real MAD-X (cpymad) on the source, comparing per-element optics through
:func:`lattix.oracles.compare.compare_pair`.

The charge-sign phase rule is pinned by *tracking*, not by a formula: the same
IR cavity at ``phi_s = -30 deg`` is written for a proton and for an H⁻ and both
decks must gain ``+V·cos30`` in elegant's own reference momentum.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix.formats.elegant import Reader, Writer
from lattix.formats.madx import Reader as MadxReader
from lattix.ir.elements import RFP, Bend, BendP, Drift, RFCavity
from lattix.ir.lattice import Lattice
from lattix.ir.reference import ReferenceParticle, species
from lattix.oracles import get_oracle
from lattix.oracles.base import BeamSpec
from lattix.oracles.compare import compare_pair
from lattix.testing import needs, require

require("elegant")
pytestmark = pytest.mark.oracle_elegant

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "helix"
TOL = 1e-8


def beam_of(lat: Lattice) -> BeamSpec:
    return BeamSpec(species=lat.reference.species.name,
                    kinetic_energy_eV=lat.reference.kinetic_energy_eV)


@needs("elegant")
@needs("madx")
@pytest.mark.oracle_madx
def test_fodo_written_as_lte_matches_madx_on_the_source(tmp_path):
    """A2 elegant leg (Exact tier): fodo.madx has no RF, so the two engines must agree
    on the *cumulative* maps to 1e-8.  Measured 2026-09-03: T4x4 2.8e-10, disp 1.2e-15."""
    src = DATA / "fodo.madx"
    lat, rep_in = MadxReader().read(src)
    out = tmp_path / "fodo.lte"
    rep_out = Writer().write(lat, out, strict=True)
    assert rep_in.ok and rep_out.ok

    a = get_oracle("madx").run(src, workdir=tmp_path / "a")
    b = get_oracle("elegant").run(out, beam=beam_of(lat), workdir=tmp_path / "b")
    cmp = compare_pair(a, b)

    assert cmp.n_shared >= 8
    assert cmp.length_a == pytest.approx(cmp.length_b, abs=1e-12)
    assert cmp.length_b == pytest.approx(lat.total_length, abs=1e-12)
    assert cmp.blocks["T4x4"] < TOL, cmp.row()
    assert cmp.blocks["disp"] < TOL, cmp.row()
    assert cmp.max_rcum_abs < TOL, cmp.row()
    assert [n.upper() for n in a.names] == [n.upper() for n in b.names]
    assert cmp.notes == []


@needs("elegant")
@needs("madx")
@pytest.mark.oracle_madx
def test_transport_written_as_lte_matches_madx_element_by_element(tmp_path):
    """transport.madx carries a 0.4 m cavity at 800 MeV (beta = 0.8418), where elegant's
    RFCA matrix is only ultra-relativistically correct: its R65 is exactly ``beta`` times
    the physical value (PLAN §8, docs/oracles.md).  So the *per-element* maps are compared
    at the Exact tier everywhere except the cavity, whose R65 ratio is asserted to be beta
    — the discrepancy is a known engine convention, not a translation error.

    Measured 2026-09-03: every non-RF element agrees to 2.5e-10; the cavity's R65 ratio is
    0.8418106784 against beta = 0.8418106885."""
    src = DATA / "transport.madx"
    lat, _ = MadxReader().read(src)
    out = tmp_path / "transport.lte"
    Writer().write(lat, out, strict=True)

    a = get_oracle("madx").run(src, workdir=tmp_path / "a").to_common()
    b = get_oracle("elegant").run(out, beam=beam_of(lat), workdir=tmp_path / "b").to_common()
    assert [n.upper() for n in a.names] == [n.upper() for n in b.names]
    assert a.total_length == pytest.approx(b.total_length, abs=1e-12)

    for i, name in enumerate(a.names):
        d = float(np.max(np.abs(a.R_elem[i][:4, :4] - b.R_elem[i][:4, :4])))
        if name.upper() == "CAV":
            continue
        assert d < TOL, f"{name}: per-element T4x4 differs by {d:.3e}"
    i = [n.upper() for n in a.names].index("CAV")
    ratio = b.R_elem[i][5, 4] / a.R_elem[i][5, 4]
    assert ratio == pytest.approx(lat.reference.beta, rel=1e-7)


@needs("elegant")
def test_elegant_round_trip_reproduces_itself(tmp_path):
    """.lte -> IR -> .lte must give elegant the same machine, element for element."""
    src = tmp_path / "src.lte"
    src.write_text("\n".join([
        "% 0.3 sto LQ",
        "QF: KQUAD, L=LQ, K1=0.6",
        "QD: KQUAD, L=LQ, K1=-0.6",
        "D: DRIF, L=0.7",
        "B: CSBEND, L=1.0, ANGLE=0.1, E1=0.05, E2=0.05, FINT=0",
        "S: KSEXT, L=0.2, K2=1.5",
        "C: RCOL, L=0.05, X_MAX=0.02, Y_MAX=0.01",
        "M: MARK",
        "CELL: LINE=(QF,D,B,D,QD,D,S,C,M)",
        "USE, CELL",
    ]))
    beam = BeamSpec(species="proton", kinetic_energy_eV=8e8)
    lat, rep = Reader().read(src, species=beam.species, kinetic_energy_eV=beam.kinetic_energy_eV)
    out = tmp_path / "out.lte"
    Writer().write(lat, out)

    a = get_oracle("elegant").run(src, beam=beam, workdir=tmp_path / "a")
    b = get_oracle("elegant").run(out, beam=beam, workdir=tmp_path / "b")
    cmp = compare_pair(a, b)
    assert cmp.n_shared >= 6
    assert cmp.length_a == pytest.approx(cmp.length_b, abs=1e-12)
    assert cmp.max_rcum_abs < 1e-12, cmp.row()


@needs("elegant")
@pytest.mark.parametrize("sp", ["proton", "h-"])
def test_rfca_phase_sign_rule_is_pinned_by_elegants_own_tracking(sp, tmp_path):
    """A thin 1 MV cavity at phi_s = -30 deg must gain +V*cos30 for BOTH a positive and a
    negative species — the whole point of the charge-signed crest.  Measured 2026-09-03:
    866025.4 eV for both, with elegant writing PHASE = -120 for the proton and +60 for
    the H⁻ (mod 360)."""
    ke = 2.1e6
    ref = ReferenceParticle(species=species(sp), kinetic_energy_eV=ke)
    lat = Lattice.from_sequence("CAVLINE", [
        Drift(name="D0", length=1e-3),
        RFCavity(name="CAV", length=0.0,
                 rf=RFP(voltage_V=1e6, phase_rad=-math.pi / 6, frequency_Hz=325e6)),
        Drift(name="D1", length=1e-3),
    ], ref)
    out = tmp_path / f"cav_{sp.replace('-', 'm')}.lte"
    Writer().write(lat, out, strict=True)

    r = get_oracle("elegant").run(out, beam=beam_of(lat), workdir=tmp_path / sp.replace("-", "m"))
    i = [n.upper() for n in r.names].index("CAV")
    gain = float(r.ref_kinetic_eV_out[i] - r.ref_kinetic_eV_in[i])
    assert gain == pytest.approx(1e6 * math.cos(math.radians(30)), rel=1e-7)

    written = next(ln for ln in out.read_text().splitlines() if ln.startswith("CAV:"))
    phase = float(written.split("PHASE=")[1].split(",")[0])
    assert phase % 360 == pytest.approx((-30.0 + (-90.0 if ref.species.charge > 0 else 90.0))
                                        % 360.0)


@needs("elegant")
def test_the_two_species_decks_differ_only_in_the_phase(tmp_path):
    """Same IR cavity, two reference species: elegant must see the same energy gain from
    two *different* PHASE numbers."""
    gains, phases = {}, {}
    for sp in ("proton", "h-"):
        ref = ReferenceParticle(species=species(sp), kinetic_energy_eV=2.1e6)
        lat = Lattice.from_sequence("L", [Drift(name="D0", length=1e-3),
                                          RFCavity(name="CAV", rf=RFP(voltage_V=1e6,
                                                                      phase_rad=-math.pi / 6,
                                                                      frequency_Hz=325e6))], ref)
        out = tmp_path / f"{sp.replace('-', 'm')}.lte"
        Writer().write(lat, out)
        r = get_oracle("elegant").run(out, beam=beam_of(lat),
                                      workdir=tmp_path / f"w{sp.replace('-', 'm')}")
        i = [n.upper() for n in r.names].index("CAV")
        gains[sp] = float(r.ref_kinetic_eV_out[i] - r.ref_kinetic_eV_in[i])
        phases[sp] = float(next(ln for ln in out.read_text().splitlines()
                                if ln.startswith("CAV:")).split("PHASE=")[1].split(",")[0])
    want = 1e6 * math.cos(math.radians(30))
    assert gains["proton"] == pytest.approx(want, rel=1e-7)
    assert gains["h-"] == pytest.approx(want, rel=1e-7)
    assert (phases["h-"] - phases["proton"]) % 360 == pytest.approx(180.0)


@needs("elegant")
def test_rben_chord_length_matches_elegants_own_arc(tmp_path):
    """The IR keeps the ARC length; the writer emits the CHORD.  elegant's path length
    for the written RBEN must come back as the IR arc."""
    arc = 1.0 * 0.05 / math.sin(0.05)                     # chord 1 m, angle 0.1 rad
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=8e8)
    lat = Lattice.from_sequence("BLINE", [
        Bend(name="B", length=arc, bend=BendP(angle=0.1, e1=0.05, e2=0.05, rect=True)),
    ], ref)
    out = tmp_path / "rben.lte"
    Writer().write(lat, out)
    assert "RBEN, L=1" in out.read_text()

    r = get_oracle("elegant").run(out, beam=beam_of(lat), workdir=tmp_path / "w")
    assert float(r.total_length) == pytest.approx(arc, abs=1e-12)
    assert float(r.total_length) == pytest.approx(lat.total_length, abs=1e-12)


@needs("elegant")
def test_sector_bend_written_from_the_ir_keeps_its_arc_length(tmp_path):
    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=8e8)
    lat = Lattice.from_sequence("SLINE", [
        Bend(name="B", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05)),
    ], ref)
    out = tmp_path / "sben.lte"
    Writer().write(lat, out)
    r = get_oracle("elegant").run(out, beam=beam_of(lat), workdir=tmp_path / "w")
    assert float(r.total_length) == pytest.approx(1.0, abs=1e-12)


@needs("elegant")
@needs("madx")
@pytest.mark.oracle_madx
def test_fodo_twiss_agrees_between_madx_and_the_written_lte(tmp_path):
    """golden.yaml pins fodo.madx at betx_end 7.1635 / bety_end 15.2867 in cpymad."""
    lat, _ = MadxReader().read(DATA / "fodo.madx")
    out = tmp_path / "fodo.lte"
    Writer().write(lat, out)
    a = get_oracle("madx").run(DATA / "fodo.madx", workdir=tmp_path / "a")
    b = get_oracle("elegant").run(out, beam=beam_of(lat), workdir=tmp_path / "b")
    for key in ("betx", "bety", "alfx", "alfy"):
        assert b.twiss[key][-1] == pytest.approx(a.twiss[key][-1], rel=1e-8, abs=1e-9), key
    assert b.twiss["betx"][-1] == pytest.approx(7.1635, rel=1e-4)
    assert b.twiss["bety"][-1] == pytest.approx(15.2867, rel=1e-4)
