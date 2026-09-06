"""Before/after semantics: alignment of the written deck read back with the source, per-element diffs
classified against the ledger, and the battery's round-trip verdict."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lattix import crossval
from lattix.crossval import _read_options, _suffix
from lattix.fidelity import FidelityReport
from lattix.formats import read, write
from lattix.ir.elements import RFP, Drift, Quadrupole, RFCavity
from lattix.ir.elements import MagneticMultipoleP as MP
from lattix.ir.lattice import Lattice
from lattix.ir.walk import propagate
from lattix.ui.compare import _Resolver, compare_translation, element_diffs
from tests.formats.test_ocelot_writer import ref_at

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "compare"
LW = {"species": "proton", "kinetic_energy_eV": 20e6, "frequency_Hz": 352.2e6}


def _translate(rel: str, src: str, dst: str, tmp_path: Path, opts: dict | None = None):
    lat, _ = read(DATA / rel, src, **(opts or {}))
    out = tmp_path / ("ImpactT.in" if dst == "impactt" else "ImpactZ.in" if dst == "impactz" else f"x{_suffix(dst)}")
    rep = write(lat, out, dst)
    lat2, rep2 = read(out, dst, **_read_options(dst, lat.reference.species))
    return lat, rep, lat2, rep2, compare_translation(lat, rep, lat2, rep2, src_fmt=src, dst_fmt=dst)


def _no_defects(c: dict) -> None:
    bad = [(c["alignment"]["elements"][d["i"]]["name"], d["quantities"], d["energy"]["class"], d["position"]["class"])
           for d in c["diffs"] if d["worst"] == "DIFF"]
    assert not bad, bad[:3]


@pytest.mark.parametrize("dst", ["opal", "impactt", "dynac", "elegant"])
def test_fodo_aligns_by_name_to_every_target(tmp_path, dst):
    lat, rep, lat2, rep2, c = _translate("helix/fodo.madx", "madx", dst, tmp_path)
    al = c["alignment"]
    assert all(e["match"] != "none" or e["dropped"] or e["absent"] for e in al["elements"])
    assert al["counts"]["name"] >= 7 and c["ir"]["ok"]
    occ = [e["occurrence"] for e in al["elements"] if e["name"] == "b1"]
    assert occ == [1, 2]                                   # the repeated bend keeps its order
    for t in al["targets"]:
        if t["role"] in ("primary", "part", "placeholder"):
            assert t["src"], t
        if t["role"] == "padding":
            assert t["kind"] == "Drift"
    _no_defects(c)


@pytest.mark.parametrize("dst", ["madx", "xtrack", "elegant", "impactz", "impactt"])
def test_mebt_thin_gaps_align_with_their_surrogates(tmp_path, dst):
    pytest.importorskip("cpymad") if dst == "madx" else None
    pytest.importorskip("xtrack") if dst == "xtrack" else None
    lat, rep, lat2, rep2, c = _translate("helix/mebt_line.dat", "tracewin", dst, tmp_path)
    al = c["alignment"]
    gaps = [e for e in al["elements"] if e["kind"] == "RFCavity"]
    assert gaps and all(e["match"] == "name" for e in gaps)
    for e in gaps:
        d = c["diffs"][e["i"]]
        assert d["quantities"]["gain"]["class"] == "equal" and d["quantities"]["volt"]["class"] == "equal"
    if dst in ("madx", "xtrack", "elegant"):
        lenses = [t for t in al["targets"] if t["role"] == "rf_focusing"]
        assert lenses and all(t["src"] for t in lenses)
    _no_defects(c)
    assert c["ir"]["ok"] and c["tier"] in ("exact", "equivalent", "lossy")


@pytest.mark.parametrize("dst", ["opal", "impactz"])
def test_lightwin_field_maps_align_to_their_clusters(tmp_path, dst):
    lat, rep, lat2, rep2, c = _translate("lightwin/example.dat", "tracewin", dst, tmp_path, LW)
    al = c["alignment"]
    maps = [e for e in al["elements"] if e["kind"] == "FieldMap"]
    assert len(maps) == 142 and all(e["match"] == "name" and e["dst"] for e in maps)
    kinds = {al["targets"][j]["kind"] for e in maps for j in e["dst"]}
    assert kinds <= {"RFCavity", "FieldMap", "Drift", "Solenoid", "Quadrupole"}
    for e in maps:
        d = c["diffs"][e["i"]]
        assert d["quantities"]["length"]["class"] in ("equal", "explained")
        assert d["quantities"]["gain"]["class"] in ("equal", "explained")
    _no_defects(c)


def test_thick_cavity_written_thin_absorbs_the_half_drifts(tmp_path):
    lat, rep, lat2, rep2, c = _translate("flame/LS1.lat", "flame", "tracewin", tmp_path)
    cav = next(e for e in c["alignment"]["elements"] if e["kind"] == "RFCavity")
    roles = set(cav["roles"].values())
    assert "padding" in roles and c["diffs"][cav["i"]]["quantities"]["length"]["class"] == "equal"
    _no_defects(c)


def test_resolver_strips_derived_suffixes_but_keeps_literal_names():
    lat = Lattice.from_sequence("t", [Drift(name="Q", length=0.1), Drift(name="Q_2", length=0.1),
                                      Drift(name="X", length=0.1), Drift(name="gap", length=0.1)], ref_at(sp="proton"))
    r = _Resolver(propagate(lat))
    assert r.resolve("Q_2") == ("Q_2", False) and r.resolve("Q_3") == ("Q", True)
    assert r.resolve("x_rfdefocus_2") == ("X", True) and r.resolve("X_D1") == ("X", True)
    assert r.resolve("x_k") == ("X", True) and r.resolve("nothing") == (None, False)


def _two_quads(g2: float, skew: float = 0.0):
    return Lattice.from_sequence("t", [
        Drift(name="d", length=0.5),
        Quadrupole(name="q", length=0.2, multipole=MP(Bn={1: 1.0}, Bs={1: skew})),
        Quadrupole(name="p", length=0.2, multipole=MP(Bn={1: g2})),
    ], ref_at(sp="proton"))


def test_quantity_classes_and_diff_escalation():
    a, b = _two_quads(2.0, skew=0.3), _two_quads(2.5, skew=0.0)
    rep_w = FidelityReport(target_format="x")
    rep_w.lossy("SKEW_COMPONENT_DROPPED", "skew lost", element="q", kind="Quadrupole")
    rep_w.equivalent("SOMETHING_EQUIVALENT", "model", element="p", kind="Quadrupole")
    rep_r = FidelityReport(source_format="x")
    c = compare_translation(a, rep_w, b, rep_r, src_fmt="a", dst_fmt="b")
    dq = next(d for d in c["diffs"] if c["alignment"]["elements"][d["i"]]["name"] == "q")
    dp = next(d for d in c["diffs"] if c["alignment"]["elements"][d["i"]]["name"] == "p")
    assert dq["quantities"]["BsL1"]["class"] == "explained"
    assert dq["quantities"]["BsL1"]["codes"] == ["SKEW_COMPONENT_DROPPED"]
    assert dq["worst"] == "LOSSY"
    assert dp["quantities"]["BnL1"]["class"] == "unexplained" and dp["worst"] == "DIFF"   # EQUIVALENT never explains
    assert c["summary"]["worst_counts"]["DIFF"] == 1 and c["summary"]["n_unexplained"] == 1


def test_energy_explained_by_constant_p0(tmp_path):
    pytest.importorskip("cpymad")
    lat = Lattice.from_sequence("t", [
        Drift(name="d1", length=0.5),
        RFCavity(name="c", length=0.0, rf=RFP(voltage_V=1e6, phase_rad=0.0, frequency_Hz=352.2e6)),
        Quadrupole(name="q", length=0.2, multipole=MP(Bn={1: 1.0})),
        Drift(name="d2", length=0.5),
    ], ref_at(sp="proton", f=352.2e6))
    out = tmp_path / "c.madx"
    rep = write(lat, out, "madx")
    lat2, rep2 = read(out, "madx", species="proton")
    c = compare_translation(lat, rep, lat2, rep2, src_fmt="lattix", dst_fmt="madx")
    assert c["settings"]["skip_energy"]
    dq = next(d for d in c["diffs"] if c["alignment"]["elements"][d["i"]]["name"] == "q")
    assert dq["energy"]["class"] in ("equal", "explained") and dq["worst"] != "DIFF"


def test_alignment_golden_fodo_to_madx(tmp_path):
    pytest.importorskip("cpymad")
    lat, rep, lat2, rep2, c = _translate("helix/fodo.madx", "madx", "madx", tmp_path)
    doc = crossval._round_json(json.loads(json.dumps({"alignment": c["alignment"], "diffs": c["diffs"],
                                                       "tier": c["tier"], "ir": c["ir"], "summary": c["summary"]})))
    text = json.dumps(doc, indent=1, sort_keys=True)
    golden = GOLDEN / "fodo.madx.to.madx.compare.json"
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(), "golden differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def test_element_diffs_handles_dropped_and_absent(tmp_path):
    lat, rep, lat2, rep2, c = _translate("helix/mebt_line.dat", "tracewin", "elegant", tmp_path)
    al = c["alignment"]
    gone = [e for e in al["elements"] if e["match"] == "none"]
    assert gone and all(e["dropped"] or e["absent"] for e in gone)
    for e in gone:
        d = c["diffs"][e["i"]]
        assert d["worst"] != "DIFF" and d["position"]["class"] == "equal"
    assert element_diffs(propagate(lat), propagate(lat2), al, rep, rep2,
                         crossval.roundtrip_settings(rep, rep2), lat, lat2)
