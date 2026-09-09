"""Lockstep anchor (PLAN §5.2): the same TraceWin deck reaches the IR by two code
paths that share nothing — lattix's TraceWin reader vs HELIX's parser + the HELIX
adapter — and the two IRs must agree element by element."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix.ir import Bend, Kicker, Quadrupole, RFCavity, Solenoid, propagate
from lattix.testing import helix_path, needs

pytestmark = [pytest.mark.oracle_helix, needs("helix")]

PUBLIC = Path(__file__).parents[1] / "data" / "public" / "helix"
HELIX_EXAMPLES = helix_path('examples')
DECKS = [PUBLIC / n for n in ("fodo_cell.dat", "bend_line.dat", "dtl_section.dat", "solenoid_channel.dat",
                              "mebt_line.dat")] + \
        [HELIX_EXAMPLES / "pipii" / "mebt" / "mebt.dat", HELIX_EXAMPLES / "pipii" / "mebt+hwr" / "mebt+hwr.dat"]


def _placed(lat):
    """Physical elements only (directives/markers differ legitimately between the paths)."""
    return [p for p in propagate(lat) if p.element.kind in
            ("Drift", "Quadrupole", "Solenoid", "Bend", "RFCavity", "FieldMap", "NCells", "Kicker", "Collimator")]


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.name)
def test_tracewin_reader_matches_helix_path(deck):
    if not deck.is_file():
        pytest.skip(f"{deck.name} not available")
    from lattix.formats.helix import read_helix_deck
    from lattix.formats.tracewin import Reader

    species, ke = ("h-", 2.1e6) if "pipii" in str(deck) else ("proton", 2.1e6)
    if deck.name == "dtl_section.dat":
        species, ke = "proton", 2.1e6
    lat_a, rep_a = Reader().read(deck, species=species, kinetic_energy_eV=ke)
    lat_b, rep_b = read_helix_deck(deck, species=species, kinetic_energy_eV=ke)
    pa, pb = _placed(lat_a), _placed(lat_b)
    assert [p.element.kind for p in pa] == [p.element.kind for p in pb]
    for a, b in zip(pa, pb, strict=True):
        ea, eb = a.element, b.element
        assert ea.length == pytest.approx(eb.length, abs=1e-12), ea.name
        if isinstance(ea, Quadrupole):
            assert ea.gradient == pytest.approx(eb.gradient, rel=1e-12)
            assert ea.skew_rad == pytest.approx(eb.skew_rad, abs=1e-12)
        elif isinstance(ea, Solenoid):
            assert ea.solenoid.Bsol_T == pytest.approx(eb.solenoid.Bsol_T, rel=1e-12)
        elif isinstance(ea, Bend):
            assert ea.bend.angle == pytest.approx(eb.bend.angle, abs=1e-12)
            assert ea.bend.e1 == pytest.approx(eb.bend.e1, abs=1e-12)
            assert ea.bend.e2 == pytest.approx(eb.bend.e2, abs=1e-12)
            assert ea.multipole.Bn.get(1, 0.0) == pytest.approx(eb.multipole.Bn.get(1, 0.0), rel=1e-7, abs=1e-12)
        elif isinstance(ea, RFCavity):
            assert ea.rf.voltage_V == pytest.approx(eb.rf.voltage_V, rel=1e-12)
            assert math.cos(ea.rf.phase_rad) == pytest.approx(math.cos(eb.rf.phase_rad), abs=1e-12)
            assert math.sin(ea.rf.phase_rad) == pytest.approx(math.sin(eb.rf.phase_rad), abs=1e-12)
            assert ea.rf.frequency_Hz == pytest.approx(eb.rf.frequency_Hz, rel=1e-12)
        elif isinstance(ea, Kicker):
            # ∫B·dl/Bρ: HELIX's particle masses differ from CODATA-2018 at the 1e-8 level
            assert ea.hkick == pytest.approx(eb.hkick, rel=1e-7, abs=1e-15)
            assert ea.vkick == pytest.approx(eb.vkick, rel=1e-7, abs=1e-15)
        # reference energy at each element must agree wherever the walk knows the gain
        if a.ref_in.kinetic_energy_eV and b.ref_in.kinetic_energy_eV:
            if ea.kind not in ("FieldMap", "NCells") and "FieldMap" not in {p.element.kind for p in pa[:pa.index(a)]}:
                assert a.ref_in.kinetic_energy_eV == pytest.approx(b.ref_in.kinetic_energy_eV, rel=1e-9)
    assert pa[-1].s_out == pytest.approx(pb[-1].s_out, abs=1e-9)
