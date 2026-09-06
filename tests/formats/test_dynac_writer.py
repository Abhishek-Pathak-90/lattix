"""DYNAC writer: rules coverage, the deck, the measured conventions (cm, kG, MV; charge-signed buncher
phase; negative field for negative species), the FIELD side file, goldens."""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from lattix import read, write
from lattix.formats.base import check_rules_coverage
from lattix.formats.dynac import Writer
from lattix.formats.dynac.cards import parse_deck
from lattix.ir.elements import RFP, Drift, Octupole, RFCavity
from lattix.ir.lattice import Lattice
from tests.formats.test_ocelot_writer import all_kinds_lattice, ref_at

DATA = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "dynac"


def _deck(lat: Lattice, tmp_path: Path, **opts):
    out = tmp_path / "lat.dynac.in"
    rep = write(lat, out, "dynac", **opts)
    title, cards = parse_deck(out.read_text(encoding="latin-1"))
    return title, cards, rep, out


def _by_name(cards):
    """``{element name: [cards]}`` from the ``; lattix:`` tags (a tag holds until the next one)."""
    groups: dict[str, list] = {}
    cur = None
    for c in cards:
        if c.name == "STOP":
            break
        for ln in c.comments:
            if "lattix: name=" in ln:
                cur = ln.split("name=")[1].split()[0].strip('"')
        if cur is not None:
            groups.setdefault(cur, []).append(c)
    return groups


def _golden(name: str, path: Path) -> None:
    golden = GOLDEN / name
    text = path.read_text(encoding="latin-1")
    if os.environ.get("LATTIX_UPDATE_GOLDEN") or not golden.exists():
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(text)
    assert text == golden.read_text(encoding="latin-1"), f"golden {name} differs (LATTIX_UPDATE_GOLDEN=1 to regenerate)"


def test_rules_cover_every_kind():
    assert check_rules_coverage(Writer()) == set()


def test_all_kinds_deck_cards_and_ledger(tmp_path):
    lat = all_kinds_lattice("proton")
    title, cards, rep, out = _deck(lat, tmp_path)
    assert title.startswith("all_kinds")
    names = [c.name for c in cards]
    assert names[:3] == ["GEBEAM", "INPUT", "EMIPRT"] and names[-1] == "STOP"
    g = _by_name(cards)
    assert [c.name for c in g["q1"]] == ["ALINER", "QUADRUPO", "ALINER"]
    assert [c.name for c in g["b1"]] == ["BMAGNET"] and [c.name for c in g["sol1"]] == ["SOLENO"]
    assert [c.name for c in g["c1"]] == ["BUNCHER"] and [c.name for c in g["c2"]] == ["DRIFT", "BUNCHER", "DRIFT"]
    assert [c.name for c in g["k1"]] == ["STEER", "STEER", "DRIFT"] and [c.name for c in g["m1"]] == ["STEER", "STEER"]
    assert [c.name for c in g["col1"]] == ["REJECT", "DRIFT", "REJECT"] and [c.name for c in g["f1"]] == ["STRIPPER"]
    assert [c.name for c in g["fq1"]] == ["NEWF"] and [c.name for c in g["rc1"]] == ["NREF"]
    assert [c.name for c in g["d4"]] == ["REJECT", "DRIFT", "REJECT"] and [c.name for c in g["mk1"]] == ["EMIT"]
    assert not (tmp_path / "lat.fields.txt").exists()          # no FIELD block: thick cavities are bunchers
    codes = rep.codes()
    for code in ("OCTUPOLE_TO_DRIFT", "MULTIPOLE_AS_STEER", "THICK_CAVITY_AS_BUNCHER", "COLLIMATOR_AS_REJECT",
                 "FOIL_AS_STRIPPER", "TAYLOR_DROPPED", "PATCH_AS_ALINER", "REFCHANGE_AS_NREF", "FOREIGN_DIRECTIVE",
                 "APERTURE_AS_REJECT", "MISALIGN_AS_ALINER", "INSTRUMENT_AS_EMIT", "KICKER_THIN_AT_ENTRANCE"):
        assert code in codes, code
    assert "; lattix: reference species=\"proton\"" in out.read_text(encoding="latin-1")


def test_measured_conventions(tmp_path):
    lat = all_kinds_lattice("proton")
    title, cards, _, _ = _deck(lat, tmp_path)
    g = _by_name(cards)
    brho = lat.reference.brho_signed
    # INPUT: rest mass MeV, mass number, charge; kinetic energy MeV
    inp = next(c for c in cards if c.name == "INPUT")
    assert inp.floats(0)[0] == pytest.approx(938.27208816) and inp.floats(0)[2] == 1.0
    assert inp.floats(1)[0] == pytest.approx(2.1)
    q = next(c for c in g["q1"] if c.name == "QUADRUPO").floats(0)
    assert q[0] == 20.0 and q[2] == 5.0 and q[1] == pytest.approx(1.2 * 0.05 * 10.0)     # kG at 5 cm for 1.2 T/m
    b = g["b1"][0]
    assert b.floats(1)[:3] == pytest.approx([math.degrees(0.1), 1000.0, 0.0])
    assert b.floats(2) == pytest.approx([math.degrees(0.05), 0.0, 0.45, 0.0, 3.0])
    assert g["sol1"][0].floats(0) == pytest.approx([1.0, 30.0, 5.0])                    # 0.5 T = 5 kG
    c1 = g["c1"][0].floats(0)
    assert c1 == pytest.approx([0.08, -85.0, 1.0, 5.0])                                  # MV, deg (= φs), harmonic
    cav = next(c for c in g["c2"] if c.name == "BUNCHER")
    assert cav.floats(0) == pytest.approx([1.0, -30.0, 1.0, 5.0]) and g["c2"][0].floats(0) == [10.0]
    st = [c.floats(0) for c in g["k1"] if c.name == "STEER"]
    ref_k1 = next(p.ref_in for p in __import__("lattix.ir.walk", fromlist=["propagate"]).propagate(lat)
                  if p.element.name == "k1")
    assert st[0] == pytest.approx([1e-3 * ref_k1.brho_signed, 0]) and st[1][1] == 1
    m = [c.floats(0) for c in g["m1"]]
    assert m[0] == pytest.approx([-0.01, 0]) and m[1] == pytest.approx([0.02, 1])       # STEER FLD = ∓BnL0 / BsL0 [T·m]
    assert g["col1"][0].floats(0) == pytest.approx([1, 1000.0, 4000.0, 1.0, 2.0, 400.0])
    assert g["f1"][0].floats(0) == pytest.approx([6, 12.011, 1e-4, 1])                  # C, 1e-3 kg/m² = 1e-4 g/cm²
    assert g["rc1"][0].floats(0) == pytest.approx([0.0, 1e-3, 0, 1])
    assert brho > 0


def test_negative_species_flips_buncher_phase_and_bend_field(tmp_path):
    lat = all_kinds_lattice("h-")
    _, cards, rep, _ = _deck(lat, tmp_path)
    g = _by_name(cards)
    assert g["c1"][0].floats(0)[1] == pytest.approx(95.0)                    # −85° + 180°: gain = q·V·cos φ
    ref_b1 = next(p.ref_in for p in __import__("lattix.ir.walk", fromlist=["propagate"]).propagate(lat)
                  if p.element.name == "b1")
    baim = g["b1"][0].floats(1)[2]
    assert baim == pytest.approx(-ref_b1.brho_abs / 10.0 * 10.0)             # −|Bρ|/ρ in kG for ρ = 10 m
    q = next(c for c in g["q1"] if c.name == "QUADRUPO").floats(0)
    assert q[1] == pytest.approx(1.2 * 0.05 * 10.0)                          # the lab field, whatever the charge
    assert "OCELOT_ELECTRON_ONLY" not in rep.codes()


def test_field_file_blocks(tmp_path):
    """An NCells train is a FIELD block (frequency line, z/E pairs over the nonzero span, terminator)."""
    from lattix.ir.elements import NCells

    lat = Lattice.from_sequence("t", [Drift(name="d", length=0.1),
                                      NCells(name="n", length=0.3, rf=RFP(voltage_V=3e5, phase_rad=-0.5,
                                                                          frequency_Hz=325e6, n_cell=3),
                                             params={"n_cells": 3, "mode": 1})],
                                ref_at(sp="proton", ke=20e6))
    _, cards, rep, out = _deck(lat, tmp_path)
    field = (tmp_path / "lat.fields.txt").read_text().splitlines()
    assert field[0] == "325000000." and field[-1] == "0. 0."
    z = [float(ln.split()[0]) for ln in field[1:-1]]
    e = [float(ln.split()[1]) for ln in field[1:-1]]
    assert z[0] == 0.0 and z[-1] == pytest.approx(0.3, abs=2e-3) and max(e) > 0 and min(e) < 0
    assert "NCELLS_AS_CAVNUM" in rep.codes()
    assert "V=300000" in out.read_text(encoding="latin-1")
    assert [c.name for c in _by_name(cards)["n"]] == ["FIELD", "CAVNUM"]


def test_thin_gap_of_a_second_frequency_is_a_harmonic_buncher(tmp_path):
    lat = Lattice.from_sequence("t", [Drift(name="d", length=0.1),
                                      RFCavity(name="c", rf=RFP(voltage_V=1e5, phase_rad=0.0, frequency_Hz=325e6))],
                                ref_at(sp="proton", f=162.5e6))
    _, cards, rep, _ = _deck(lat, tmp_path)
    assert _by_name(cards)["c"][0].floats(0) == pytest.approx([0.1, 0.0, 2.0, 5.0])


def test_strict_mode_raises_on_a_lossy_kind(tmp_path):
    from lattix.fidelity import TranslationError

    lat = Lattice.from_sequence("s", [Octupole(name="o", length=0.1)], ref_at())
    with pytest.raises(TranslationError):
        write(lat, tmp_path / "o.dynac.in", "dynac", strict=True)


def test_goldens(tmp_path):
    pytest.importorskip("cpymad")
    lat, _ = read(DATA / "helix" / "fodo.madx", "madx")
    out = tmp_path / "fodo.dynac.in"
    write(lat, out, "dynac")
    _golden("fodo.dynac.in", out)
    out2 = tmp_path / "all_kinds.dynac.in"
    write(all_kinds_lattice(), out2, "dynac")
    _golden("all_kinds.dynac.in", out2)
