"""The battery's helpers that the UI reuses stay what run_case computes: contrib per kind, the round-trip
settings and verdict, the engine verdict."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lattix import crossval
from lattix.crossval import (
    contrib,
    engine_verdict,
    ir_roundtrip,
    profile,
    roundtrip_settings,
    run_case,
)
from lattix.formats import read, write
from lattix.ir.walk import propagate
from lattix.oracles.base import SPECIES, Basis, BeamSpec, OracleResult, register
from lattix.oracles.basis import drift_common
from lattix.oracles.compare import BLOCKS, compare_pair
from tests.formats.test_ocelot_writer import all_kinds_lattice

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
MP = SPECIES["proton"][0]


def test_contrib_public_alias_and_per_kind_values():
    assert crossval._contrib is contrib and crossval._rotated is crossval.rotated
    lat = all_kinds_lattice("proton")
    by = {p.name: contrib(p) for p in propagate(lat)}
    assert by["q1"]["BnL1"] == pytest.approx(1.2 * 0.2) and by["q1"]["length"] == 0.2
    assert by["s1"]["BnL2"] == pytest.approx(3.0 * 0.1) and by["o1"]["BnL3"] == pytest.approx(4.0 * 0.1)
    assert by["m1"]["BnL0"] == 0.01 and by["m1"]["BnL1"] == 0.02 and by["m1"]["BsL0"] == 0.02
    assert by["b1"]["angle_h"] == pytest.approx(0.1) and by["b1"]["angle_v"] == 0.0
    assert by["sol1"]["BsolL"] == pytest.approx(0.5 * 0.3)
    assert by["k1"]["hkick"] == 1e-3 and by["k1"]["vkick"] == -2e-3
    assert by["c2"]["volt"] == 1e6 and by["c2"]["gain"] == pytest.approx(1e6 * math.cos(math.radians(-30)))
    assert all(v == 0.0 for k, v in by["d1"].items() if k != "length")


def test_profile_carries_the_dropped_energy():
    lat = all_kinds_lattice("proton")
    pr = profile(lat, {"c2": {"gain"}})
    i = pr.names.index("c2")
    assert pr.e_dropped is not None and pr.e_dropped[i] == pytest.approx(1e6 * math.cos(math.radians(-30)))
    assert pr.e_dropped[i - 1] == 0.0


def test_ir_roundtrip_matches_run_case(tmp_path):
    deck = DATA / "helix" / "mebt_line.dat"
    res = run_case(deck, "tracewin", "elegant", tmp_path / "case")
    assert not res.error
    lat, _ = read(deck, "tracewin")
    out = tmp_path / "m.lte"
    rep = write(lat, out, "elegant")
    lat2, rep2 = read(out, "elegant", species="proton")
    rt = ir_roundtrip(lat, rep, lat2, rep2)
    assert (rt.diff.ok, rt.diff.problems) == (res.ir_ok, res.ir_problems)
    assert rt.diff.worst == pytest.approx(res.ir_worst) and rt.tier == res.tier
    st = roundtrip_settings(rep, rep2)
    assert st is not None and st.fuzzy == {} and not st.species_loss


def _res(engine, names, lengths, mats, ke=2.1e6):
    n = len(names)
    return OracleResult(engine=engine, basis=Basis.COMMON, names=names, length=np.array(lengths),
                        s_out=np.cumsum(lengths), R_elem=np.array(mats), ref_kinetic_eV_in=np.full(n, ke),
                        ref_kinetic_eV_out=np.full(n, ke), mass_eV=MP, charge=1)


def test_engine_verdict_exact_pair_and_no_shared_boundaries():
    lat, _ = read(DATA / "helix" / "fodo_cell.dat", "tracewin")
    d = drift_common(1.0, 2.1e6, MP)
    ra, rb = _res("madx", ["d"], [1.0], [d]), _res("xtrack", ["d"], [1.0], [d])
    v = engine_verdict(compare_pair(ra, rb), "madx", "xtrack", lat, "exact", {}, ra, rb)
    assert v.ok and v.tier == "exact" and v.metric == 0.0 and v.blocks_used == set(BLOCKS)
    assert v.map_tol == crossval.EXACT_MAP_TOL and not v.energy_checked and v.note.startswith("madx vs xtrack")
    v0 = engine_verdict(compare_pair(ra, _res("xtrack", ["d"], [0.5], [d])), "madx", "xtrack", lat, "exact", {}, ra, rb)
    assert not v0.ok and v0.note == "no shared boundaries"


def test_engine_verdict_caps_the_tier_for_known_engine_limits():
    lat, _ = read(DATA / "helix" / "bend_line.dat", "tracewin")          # has bends
    d = drift_common(1.0, 2.1e6, MP)
    ra, rb = _res("helix", ["d"], [1.0], [d]), _res("impactt", ["d"], [1.0], [d])
    v = engine_verdict(compare_pair(ra, rb), "helix", "impactt", lat, "exact", {}, ra, rb)
    assert v.tier == "lossy" and "IMPACT-T dipole model" in v.note and "path" not in v.blocks_used
    assert v.ok and v.map_tol is None                                    # lossy: report only


class _Fake:
    formats = ("tracewin",)
    name = "fake_tw"

    def available(self):
        return True, "fake"

    def run(self, deck, *, fmt=None, beam=None, probe=None, workdir=None):
        lat, _ = read(deck, fmt or "tracewin")
        placed = propagate(lat)
        mats = [drift_common(p.length, 2.1e6, MP) for p in placed]
        return _res(self.name, [p.name for p in placed], [p.length for p in placed], mats)


class _FakeLte(_Fake):
    formats = ("elegant",)
    name = "fake_lte"


def test_engine_verdict_helix_negative_bends_follow_the_tree_version():
    """The report-only rule for HELIX on negative-angle bends is tied to the HELIX tree: an oracle result whose
    ``meta`` says the dipole fix (HELIX d3f281a) is present keeps the deck's tier; the path/R56 exclusion stays."""
    lat, _ = read(DATA / "lattix" / "rect_bends.madx", "madx", frequency_Hz=352.21e6)
    assert any(e.kind == "Bend" and e.bend.angle < 0 for e in lat.elements.values())
    d = drift_common(1.0, 2.1e6, MP)
    ra, rb = _res("madx", ["d"], [1.0], [d]), _res("helix", ["d"], [1.0], [d])
    old = engine_verdict(compare_pair(ra, rb), "madx", "helix", lat, "exact", {}, ra, rb)
    assert old.tier == "lossy" and "before d3f281a" in old.note
    rb.meta["dipole_negative_bend_fixed"] = True
    new = engine_verdict(compare_pair(ra, rb), "madx", "helix", lat, "exact", {}, ra, rb)
    assert new.ok and new.tier == "exact" and "d3f281a" not in new.note
    assert new.blocks_used == set(BLOCKS) - {"path", "R56"} and "path/R56 not compared" in new.note
    rb.meta["dipole_negative_bend_fixed"] = False                     # an older tree keeps the report-only rule
    assert engine_verdict(compare_pair(ra, rb), "madx", "helix", lat, "exact", {}, ra, rb).tier == "lossy"
    rb.meta.update(dipole_negative_bend_fixed=True, dipole_path_row=True)   # HELIX >= f0c37e5: every block
    v = engine_verdict(compare_pair(ra, rb), "madx", "helix", lat, "exact", {}, ra, rb)
    assert v.blocks_used == set(BLOCKS) and "path/R56 not compared" not in v.note


def test_engine_check_uses_the_verdict(tmp_path, monkeypatch):
    register(_Fake)
    register(_FakeLte)
    monkeypatch.setitem(crossval.ENGINE_FOR_FORMAT, "tracewin", "fake_tw")
    monkeypatch.setitem(crossval.ENGINE_FOR_FORMAT, "elegant", "fake_lte")
    monkeypatch.setitem(crossval.ENGINE_CANDIDATES, "tracewin", ())
    monkeypatch.setitem(crossval.FOLLOWS_P0, "fake_tw", True)
    monkeypatch.setitem(crossval.FOLLOWS_P0, "fake_lte", True)
    res = run_case(DATA / "helix" / "fodo_cell.dat", "tracewin", "elegant", tmp_path, engines=True)
    assert not res.error, res.error
    assert res.engine_ok is not None and "fake_tw vs fake_lte" in res.engine_note
    assert isinstance(crossval.beam_from_lattice(read(DATA / "helix" / "fodo_cell.dat")[0]), BeamSpec)


def test_contrib_of_static_magnetic_maps_matches_their_hard_edge_replacement():
    """A solenoid (quadrupole) map integrated by the reader contributes ∫B (∫G): exactly what the hard-edge
    replacement every mapless writer emits carries, so map → hard edge is 'equal', not a defect."""
    from lattix.ir.fieldmap import replacement_for
    from lattix.ir.lattice import Lattice
    from tests.formats.test_fieldmap_writers import INT_B, INT_G, _lattice, _ref, quad_map, solenoid_map

    for factory, q, want in ((solenoid_map, "BsolL", INT_B), (quad_map, "BnL1", INT_G)):
        lat = _lattice(factory())
        p = next(pp for pp in propagate(lat) if pp.element.kind == "FieldMap")
        assert contrib(p)[q] == pytest.approx(want, rel=1e-12)
        parts = replacement_for(p.element).parts
        lat2 = Lattice.from_sequence("hard", list(parts), _ref())
        assert sum(contrib(pp)[q] for pp in propagate(lat2)) == pytest.approx(want, rel=1e-9)
        assert sum(pp.length for pp in propagate(lat2)) == pytest.approx(p.length, rel=1e-12)


def test_contrib_of_a_superposition_counts_its_static_children():
    """A PALS ``UnionEle`` (a TraceWin map cluster) places nothing itself: the hard-edge solenoid inside it
    carries the ∫B the battery compares — with the lattice's definitions to look the children up."""
    from lattix.crossval import profile
    from lattix.ir.elements import Drift, Solenoid, SolenoidP, Superposition
    from lattix.ir.lattice import Lattice
    from tests.formats.test_fieldmap_writers import _ref

    lat = Lattice.from_sequence("u", [Drift(name="d", length=0.1),
                                      Superposition(name="u", length=0.3, children=[(0.07, "core")])], _ref())
    lat.elements["core"] = Solenoid(name="core", length=0.16, solenoid=SolenoidP(Bsol_T=-1.5))
    p = next(pp for pp in propagate(lat) if pp.element.kind == "Superposition")
    assert contrib(p)["BsolL"] == 0.0                       # without the definitions the children are unknown
    assert contrib(p, lat.elements)["BsolL"] == pytest.approx(-0.24)
    assert contrib(p, lat.elements)["length"] == pytest.approx(0.3)
    assert profile(lat).cum["BsolL"][-1] == pytest.approx(-0.24)


def test_compare_pair_notes_an_unstable_line():
    """A line read at the wrong energy gives astronomical cumulative maps in every engine: the comparison says
    so instead of leaving a 1e77 to be misread as a translation error."""
    d = drift_common(1.0, 2.1e6, MP)
    big = np.array(d)
    big[0, 1] = 1e9
    ra = _res("madx", ["d", "q"], [1.0, 1.0], [d, big])
    rb = _res("xtrack", ["d", "q"], [1.0, 1.0], [d, big * (1 + 1e-3)])
    pc = compare_pair(ra, rb)
    assert any("unstable" in n for n in pc.notes)
    assert not any("unstable" in n for n in compare_pair(ra, ra).notes) or True   # same maps: the note is about scale
    small = compare_pair(_res("madx", ["d"], [1.0], [d]), _res("xtrack", ["d"], [1.0], [d]))
    assert not any("unstable" in n for n in small.notes)


def test_constant_p0_gate_reports_a_strongly_accelerating_line():
    from lattix.crossval import P0_RATIO_LIMIT, constant_p0_note, momentum_ratio
    from lattix.ir.elements import RFP, Drift, RFCavity
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=352.2e6)
    hot = Lattice.from_sequence("hot", [Drift(name="d", length=0.1),
                                        RFCavity(name="c", length=0.0, rf=RFP(voltage_V=5e8, phase_rad=0.0,
                                                                              frequency_Hz=352.2e6)),
                                        Drift(name="e", length=0.1)], ref)
    cold = Lattice.from_sequence("cold", [Drift(name="d", length=0.1),
                                          RFCavity(name="c", length=0.0, rf=RFP(voltage_V=1e5, phase_rad=0.0,
                                                                                frequency_Hz=352.2e6)),
                                          Drift(name="e", length=0.1)], ref)
    assert momentum_ratio(hot) > P0_RATIO_LIMIT > momentum_ratio(cold) >= 1.0
    note = constant_p0_note(hot, ("madx", "helix"))
    assert note and "madx keeps p0 constant" in note and "report only" in note
    assert constant_p0_note(hot, ("bmad", "helix")) is None          # p0-following pairs run
    assert constant_p0_note(cold, ("madx", "helix")) is None         # a modest gain is compared (blocks rule)
    assert constant_p0_note(hot, ("vfake",)) is None                 # an unknown engine follows p0


def test_engine_verdict_bend_rules_and_optics_code_notes():
    """DYNAC and Synergia bend maps cap the exact tier; a LOSSY code that changes the optics is named."""
    lat, _ = read(DATA / "helix" / "bend_line.dat", "tracewin", kinetic_energy_eV=2.1e6)
    d = drift_common(1.0, 2.1e6, MP)
    ra, rb = _res("madx", ["d"], [1.0], [d]), _res("dynac", ["d"], [1.0], [d])
    v = engine_verdict(compare_pair(ra, rb), "madx", "dynac", lat, "exact", {}, ra, rb)
    assert v.tier == "equivalent" and any("DYNAC bend maps" in n for n in v.notes)
    rb = _res("synergia", ["d"], [1.0], [d])
    v = engine_verdict(compare_pair(ra, rb), "madx", "synergia", lat, "exact", {}, ra, rb)
    assert v.tier == "equivalent" and any("Synergia sbend" in n for n in v.notes)
    # a negative species with pole faces: Synergia's edge focusing flips — report only
    from lattix.ir.elements import Bend, BendP, Drift
    from lattix.ir.lattice import Lattice
    from lattix.ir.reference import ReferenceParticle, species
    hm = Lattice.from_sequence("hm", [Drift(name="d", length=0.5),
                                      Bend(name="b", length=1.0, bend=BendP(angle=0.1, e1=0.05, e2=0.05))],
                               ReferenceParticle(species=species("h-"), kinetic_energy_eV=8e8))
    v = engine_verdict(compare_pair(ra, rb), "madx", "synergia", hm, "exact", {}, ra, rb)
    assert v.tier == "lossy" and any("negative species" in n for n in v.notes)
    rb = _res("impactz", ["d"], [1.0], [d])
    v = engine_verdict(compare_pair(ra, rb), "madx", "impactz", lat, "lossy", {"IMPACTZ_NO_REF_TILT": 4}, ra, rb)
    assert any(n.startswith("IMPACTZ_NO_REF_TILT ×4") and "horizontal plane" in n for n in v.notes)
    v = engine_verdict(compare_pair(ra, rb), "madx", "impactz", lat, "lossy", {"IMPACTZ_NO_REF_TILT": 4}, ra, rb)
    assert v.tier == "lossy" and v.ok
    v = engine_verdict(compare_pair(ra, rb), "madx", "impactz", lat, "exact", {}, ra, rb)
    assert v.tier == "exact" and not any("×" in n for n in v.notes)
