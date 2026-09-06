"""Derived per-element numbers and statement location for the inspector."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lattix.formats import read, write
from lattix.ir.walk import propagate
from lattix.ui.inspect import UNITS, derived_numbers, locate_statements, reference_state
from tests.formats.test_ocelot_writer import all_kinds_lattice

DATA = Path(__file__).resolve().parents[1] / "data" / "public"


def test_derived_numbers_for_every_kind_are_json_and_finite():
    lat = all_kinds_lattice("proton")
    seen = set()
    for p in propagate(lat):
        d = derived_numbers(p, lat)
        seen.add(p.element.kind)
        text = json.dumps(d, allow_nan=False, default=str)
        assert "NaN" not in text
        for k, v in d.items():
            if isinstance(v, float):
                assert math.isfinite(v), (p.name, k)
            if k not in ("map_summary", "replacement", "knl", "ksl", "gain_explicit"):
                assert k in UNITS or k in ("angle_h", "angle_v", "k2s", "k3s"), k
    assert {"Quadrupole", "Bend", "Solenoid", "RFCavity", "Kicker", "Taylor", "Foil", "Multipole"} <= seen
    placed = {p.name: p for p in propagate(lat)}
    by = {n: derived_numbers(p, lat) for n, p in placed.items()}
    ref = lat.reference
    ref_k = placed["k1"].ref_in                      # the kicker sits after the cavities
    assert by["q1"]["k1"] == pytest.approx(1.2 / ref.brho_signed) and by["q1"]["GL"] == pytest.approx(0.24)
    assert by["sol1"]["ks"] == pytest.approx(0.5 / ref.brho_signed) and by["sol1"]["int_B_Tm"] == pytest.approx(0.15)
    assert by["b1"]["rho_m"] == pytest.approx(10.0) and by["b1"]["B0_T"] == pytest.approx(ref.brho_signed * 0.1)
    assert by["c2"]["gain_eV"] == pytest.approx(1e6 * math.cos(math.radians(-30)))
    assert by["c2"]["phase_deg"] == pytest.approx(-30) and by["c1"]["rf_defocusing_1_per_m"] != 0
    assert by["k1"]["int_Bdl_x_Tm"] == pytest.approx(1e-3 * ref_k.brho_signed)
    assert by["t1"]["symplectic_err"] < 1e-12 and by["m1"]["hkick"] == pytest.approx(-0.01 / ref.brho_signed)
    assert by["f1"]["thickness_mg_cm2"] == pytest.approx(0.1)


def test_reference_state_and_field_map_summary():
    lat, _ = read(DATA / "lightwin" / "example.dat", species="proton", kinetic_energy_eV=20e6, frequency_Hz=352.2e6)
    p = next(p for p in propagate(lat) if p.element.kind == "FieldMap")
    d = derived_numbers(p, lat)
    assert d["map_summary"]["kind"] == "rf" and d["replacement"]["parts"]
    st = reference_state(p.ref_in)
    assert st["beta_lambda_m"] == pytest.approx(st["beta"] * st["wavelength_m"])
    assert st["ke_eV"] == pytest.approx(20e6, rel=0.5)


@pytest.mark.parametrize("fmt,species", [("elegant", "proton"), ("opal", None), ("impactt", None), ("dynac", None)])
def test_locate_statements_in_written_decks(tmp_path, fmt, species):
    lat, _ = read(DATA / "helix" / "mebt_line.dat")
    out = tmp_path / fmt / ("ImpactT.in" if fmt == "impactt" else f"m.{fmt}.in")
    out.parent.mkdir()
    write(lat, out, fmt)
    lines = out.read_text(encoding="latin-1").splitlines()
    hits = locate_statements(lines, fmt, "QUAD_0001")
    assert hits and hits[0]["role"] in ("tag", "definition") and "QUAD_0001" in hits[0]["text"].upper() \
        or any("QUAD_0001" in h["text"].upper() for h in hits)
    assert all(h["line"] >= 1 for h in hits)
