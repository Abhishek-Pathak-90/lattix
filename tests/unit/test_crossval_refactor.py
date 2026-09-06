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
