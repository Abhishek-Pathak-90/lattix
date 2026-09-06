"""The browser page's view models: rows, definitions, glyph hints, ledger join, option schemas, catalogues."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lattix.formats import read, write
from lattix.ui.model import (
    canonical_read_options,
    catalog_view,
    coerce_options,
    format_catalog,
    jsonable,
    lattice_view,
    option_schema,
    sample_decks,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "public"


def _view(rel: str, **opts):
    lat, rep = read(DATA / rel, **opts)
    return lattice_view(lat, rep, fmt="tracewin", path=DATA / rel), lat


def test_fodo_cell_rows_definitions_and_ledger():
    v, lat = _view("helix/fodo_cell.dat")
    assert v["header"]["n_placed"] == 24 and v["header"]["total_length"] == pytest.approx(1.6)
    assert v["header"]["counts"]["Quadrupole"] == 8 and v["header"]["format"] == "tracewin"
    assert v["header"]["reference"]["species"]["name"] == "proton" and v["header"]["reference"]["beta"] > 0
    signs = [r["glyph"]["sign"] for r in v["rows"] if r["kind"] == "Quadrupole"]
    assert signs == [1, -1] * 4 and all(r["glyph"]["k1"] for r in v["rows"] if r["kind"] == "Quadrupole")
    assert all(r["def"] in v["definitions"] for r in v["rows"])
    assert all(r["name"] in v["ledger"]["by_element"] for r in v["rows"])
    assert [r["s_in"] for r in v["rows"]] == pytest.approx([p.s_in for p in lat.flatten()])
    json.dumps(v, allow_nan=False)
    assert v["header"]["maxima"]["k1"] > 0 and v["bbox"]["s"] == [0.0, pytest.approx(1.6)]


def test_bend_line_glyphs_and_survey():
    v, _ = _view("helix/bend_line.dat")
    bends = [r for r in v["rows"] if r["kind"] == "Bend"]
    assert bends and all(r["glyph"]["shape"] == "bend" and r["glyph"]["plane"] == "h" for r in bends)
    b = bends[0]
    assert b["survey"]["out"][3] == pytest.approx(b["survey"]["in"][3] - b["glyph"]["angle"])  # θ decreases
    assert b["derived"]["rho_m"] == pytest.approx(b["L"] / b["glyph"]["angle"])
    assert v["bbox"]["x"][0] < 0 < v["bbox"]["z"][1]


def test_mebt_line_thin_gaps_and_energy():
    v, _ = _view("helix/mebt_line.dat")
    gaps = [r for r in v["rows"] if r["kind"] == "RFCavity"]
    assert gaps and all(r["glyph"]["thin"] and r["contrib"]["gain"] for r in gaps)
    assert all(r["ke_out"] != r["ke_in"] for r in gaps) and "rf_defocusing_1_per_m" in gaps[0]["derived"]
    kick = next(r for r in v["rows"] if r["kind"] == "Kicker")
    assert "hx" in kick["glyph"] and kick["derived"] == {} or "int_Bdl_x_Tm" in kick["derived"]


def test_lightwin_field_maps_payload_and_speed():
    t0 = time.perf_counter()
    v, _ = _view("lightwin/example.dat", species="proton", kinetic_energy_eV=20e6, frequency_Hz=352.2e6)
    assert time.perf_counter() - t0 < 4.0
    assert v["header"]["n_placed"] == 633 and v["header"]["counts"]["FieldMap"] == 142
    fm = next(r for r in v["rows"] if r["kind"] == "FieldMap")
    assert fm["derived"]["map_summary"]["kind"] == "rf" and fm["derived"]["replacement"]["code"]
    assert fm["glyph"]["integrated"] and fm["glyph"]["map_kind"] == "rf"
    assert len(json.dumps(v)) < 3_000_000


def test_target_view_carries_deck_lines(tmp_path):
    lat, _ = read(DATA / "helix" / "mebt_line.dat")
    out = tmp_path / "m.lte"
    write(lat, out, "elegant")
    lat2, rep2 = read(out, "elegant", species="proton")
    v = lattice_view(lat2, rep2, fmt="elegant", path=out, deck_text=out.read_text())
    quad = next(d for d in v["definitions"].values() if d["kind"] == "Quadrupole")
    assert quad["deck_lines"][0]["role"] == "definition" and "KQUAD" in quad["deck_lines"][0]["text"]
    assert quad["n_placed"] >= 1 and v["ledger"]["source_format"] == "elegant"


def test_option_schemas_and_coercion():
    w = {f["name"]: f for f in option_schema("madx")["write"]}
    assert w["energy_mode"]["choices"] == ["delta", "local", "constant"] and w["use_expressions"]["type"] == "bool"
    r = {f["name"]: f for f in option_schema("madx")["read"]}
    assert r["species"]["choices"] and r["species"].get("free") and r["sequence"]["canonical"] == "line"
    z = {f["name"]: f for f in option_schema("impactz")["write"]}
    assert z["grid"]["type"] == "json" and z["n_particles"]["type"] == "int"
    assert [f["name"] for f in option_schema("tracewin")["write"]] == ["frequency_Hz", "static_maps"]
    assert "thin_length_m" in {f["name"] for f in option_schema("opal")["write"]}      # via render(**options)
    assert coerce_options("impactz", "write", {"grid": "[32, 32, 32]", "n_particles": "500"}) == \
        {"grid": (32, 32, 32), "n_particles": 500}
    assert coerce_options("madx", "write", {"use_expressions": "false", "energy_mode": "local"}) == \
        {"use_expressions": False, "energy_mode": "local"}
    with pytest.raises(ValueError):
        coerce_options("madx", "write", {"nonsense": 1})
    warnings: list[str] = []
    assert canonical_read_options("madx", {"line": "fodo", "kinetic_energy_eV": "2.1e6"}, warnings) == \
        {"sequence": "fodo", "kinetic_energy_eV": 2.1e6}
    assert canonical_read_options("tracewin", {"line": "x", "species": "h-"}, warnings) == {"species": "h-"}
    assert warnings and "no line" in warnings[0]


def test_format_catalog_and_code_catalog():
    fc = format_catalog()
    assert set(fc["formats"]) == set(__import__("lattix.formats.base", fromlist=["FORMATS"]).FORMATS)
    assert fc["formats"]["astra"]["bridge"] and not fc["formats"]["astra"]["readable"]
    assert fc["formats"]["madx"]["rules"]["FieldMap"]["cls"] == "LOSSY" and fc["formats"]["opal"]["side_files"]
    assert fc["formats"]["tracewin"]["rules"] is None and "Quadrupole" in fc["kinds"]
    assert fc["formats"]["tracewin"]["engine"] == "helix" and fc["formats"]["tracewin"]["engine_candidates"]
    assert fc["formats"]["opal"]["engine"] is None
    cat = catalog_view()
    assert len(cat) > 500 and cat["FM_TO_CAVITY"]["cls"] in ("EQUIVALENT", "LOSSY")
    samples = sample_decks()
    assert any(s["path"] == "helix/fodo_cell.dat" and s["format"] == "tracewin" for s in samples)
    assert jsonable({"a": (1, 2), "n": float("nan")}) == {"a": [1, 2], "n": None}
