"""IMPACT-Z reader tests (PLAN §6 task 3.3).

The three vendored decks (``tests/data/public/impactz/Example{1,2,3}``, BSD, from the
IMPACT-Z distribution) exercise the header, the DTL/SolRF/CCL RF families, the Transport
dipole, the negative diagnostic codes and the ideal-cavity sentinel.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import pytest

from lattix.formats.impactz import Reader, Writer
from lattix.formats.impactz.reader import (
    TYPE_NAMES,
    Header,
    parse_deck,
    rfdata_name,
    species_from_header,
    split_data_line,
)
from lattix.ir.elements import Drift, Marker, Quadrupole
from lattix.ir.reference import ReferenceParticle, species
from lattix.ir.units import C_LIGHT

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "impactz"


def read(name: str):
    return Reader().read(DATA / name / "ImpactZ.in")


def kinds(lat) -> Counter:
    return Counter(p.element.kind for p in lat.flatten())


# ------------------------------------------------------------------- tokenizer
def test_slash_ends_the_record():
    toks, rest = split_data_line("0.1 2 3 4 5.0 / a comment")
    assert toks == ["0.1", "2", "3", "4", "5.0"]
    assert rest == "a comment"


def test_fortran_d_exponent_and_commas():
    toks, _ = split_data_line("1.3d9, 10.75d-3 2")
    assert toks == ["1.3d9", "10.75d-3", "2"]
    from lattix.formats.impactz.reader import _to_float

    assert _to_float("1.3d9") == pytest.approx(1.3e9)
    assert _to_float("10.75D-3") == pytest.approx(1.075e-2)


def test_header_needs_eleven_records():
    with pytest.raises(ValueError, match="11 header records"):
        parse_deck("1 1\n6 10 1 0 1\n")


def test_element_card_needs_four_numbers():
    head = "\n".join(["1 1", "6 1 1 0 1", "1 1 1 1 0 0 0", "3 0 0 1", "1", "0", "0",
                      "0 1 1 1 1 0 0", "0 1 1 1 1 0 0", "0 1 1 1 1 0 0",
                      "0 1e6 1e9 1 1e9 0"])
    with pytest.raises(ValueError, match="at least"):
        parse_deck(head + "\n1.0 2\n")


def test_scxl_is_lambda_over_two_pi():
    h = Header(frequency_Hz=324e6)
    assert h.scxl == pytest.approx(C_LIGHT / (2 * math.pi * 324e6))
    assert h.scxl == pytest.approx(0.147263739, rel=1e-8)
    assert Header(frequency_Hz=0.0).scxl == 0.0


def test_rfdata_file_names():
    assert rfdata_name(1) == "rfdata1.in"
    assert rfdata_name(42) == "rfdata42.in"
    assert rfdata_name(300) == "rfdata300.in"


# ------------------------------------------------------------------- Example 1
def test_example1_header_and_counts():
    lat, rep = read("Example1")
    h, cards, _ = parse_deck((DATA / "Example1" / "ImpactZ.in").read_text())
    assert (h.dim, h.np, h.flagmap, h.flagerr, h.flagdiag) == (6, 200, 2, 0, 1)
    assert (h.nx, h.ny, h.nz, h.flagbc) == (32, 32, 64, 1)
    assert h.flagdist == 19                     # reads particle.in
    assert h.kinetic_energy_eV == pytest.approx(3.3517600e6)
    assert h.frequency_Hz == pytest.approx(324e6)
    assert h.charge == pytest.approx(-1.0)
    # 9 DTL cells + 2 quadrupoles + 1 drift
    assert kinds(lat) == Counter({"NCells": 9, "Quadrupole": 2, "Drift": 1})
    assert [c.itype for c in cards[:3]] == [101, 101, 101]
    assert lat.reference.species.charge == -1
    assert "IMPACTZ_RFDATA_MISSING" in rep.codes()   # rfdata222.in is not vendored


def test_example1_dtl_columns():
    _, cards, _ = parse_deck((DATA / "Example1" / "ImpactZ.in").read_text())
    dtl = cards[0]
    assert dtl.itype == 101
    assert dtl.v(1) == pytest.approx(2503370.0)  # field scale
    assert dtl.v(2) == pytest.approx(3.24e8)     # RF frequency [Hz]
    assert dtl.v(3) == pytest.approx(222.0)      # theta0 [deg]
    assert dtl.v(4) == pytest.approx(852.0)      # file ID -> rfdata852.in
    assert dtl.v(5) == pytest.approx(6.5e-3)     # radius [m]
    assert dtl.v(6) == pytest.approx(2.04958e-2)   # quad 1 length [m]
    lat, _ = read("Example1")
    nc = lat.flatten()[0].element
    assert nc.rf.frequency_Hz == pytest.approx(3.24e8)
    assert nc.rf.phase_rad == pytest.approx(math.radians(dtl.v(3)))
    assert nc.rf.phase_is_sync is False          # theta0 is a driven RF phase
    assert nc.params["quad1_gradient_T_per_m"] == pytest.approx(dtl.v(7))


def test_example1_quadrupole():
    lat, _ = read("Example1")
    q = next(p.element for p in lat.flatten() if isinstance(p.element, Quadrupole))
    assert q.length == pytest.approx(4.50071e-2)
    assert q.gradient == pytest.approx(-19.1523)
    assert q.aperture.half_x == pytest.approx(9e-3)


# ------------------------------------------------------------------- Example 2
def test_example2_solrf_and_dumps():
    lat, rep = read("Example2")
    assert kinds(lat) == Counter({"Drift": 9, "FieldMap": 4, "Instrument": 1})
    h, _, _ = parse_deck((DATA / "Example2" / "ImpactZ.in").read_text())
    assert h.mass_eV == pytest.approx(0.511005e6)
    assert h.kinetic_energy_eV == pytest.approx(230e6)
    assert h.frequency_Hz == pytest.approx(1.3e9)
    assert h.current_A == pytest.approx(0.13)
    fm = next(p.element for p in lat.flatten() if p.element.kind == "FieldMap")
    assert fm.meta["impactz_type"] == 105        # SolRF
    assert fm.ke == pytest.approx(1668.0)        # the field scale column
    assert fm.rf.frequency_Hz == pytest.approx(1.3e9)
    assert fm.files == ["rfdata1.in"]
    assert "IMPACTZ_BEAM_CURRENT" in rep.codes()


def test_example2_phase_dump_is_an_instrument():
    lat, _ = read("Example2")
    inst = next(p.element for p in lat.flatten() if p.element.kind == "Instrument")
    assert inst.family == "PHASE_DUMP"
    assert inst.params["file"] == 1000
    assert inst.native["impactz"]["type"] == -2


# ------------------------------------------------------------------- Example 3
def test_example3_bends_and_ideal_cavities():
    lat, rep = read("Example3")
    c = kinds(lat)
    assert c["Bend"] == 4 and c["Quadrupole"] == 21 and c["NCells"] == 5
    b = next(p.element for p in lat.flatten() if p.element.kind == "Bend")
    assert b.bend.angle == pytest.approx(0.0221138265963)     # radians
    assert b.bend.hgap == pytest.approx(0.03)
    assert b.bend.e2 == pytest.approx(0.0221138265963)        # rectangular exit face
    assert b.bend.edge_int1 == pytest.approx(0.40)
    # the five 103 cards carry a negative file ID -> IMPACT-Z's ideal-cavity model
    cav = next(p.element for p in lat.flatten() if p.element.kind == "NCells")
    assert cav.rf.phase_is_sync is True
    assert cav.rf.gradient_V_per_m != 0.0
    assert cav.rf.voltage_V == pytest.approx(cav.rf.gradient_V_per_m * cav.length)
    assert "IMPACTZ_IDEAL_CAVITY" in rep.codes()


def test_example3_unmodelled_codes_are_recorded():
    lat, rep = read("Example3")
    assert "UNMODELLED_IMPACTZ_TYPE" in rep.codes()
    # -8 slice output, -41 wakefield and -52 laser heater have no IR kind but keep
    # their raw columns so an IMPACT-Z -> IMPACT-Z round trip re-emits them
    markers = [p.element for p in lat.flatten() if isinstance(p.element, Marker)]
    assert markers and all(m.native["impactz"]["passthrough"] for m in markers)
    assert {m.native["impactz"]["type"] for m in markers} == {-8, -41, -52}


def test_example3_total_length():
    lat, _ = read("Example3")
    assert lat.total_length == pytest.approx(73.4217895, rel=1e-8)


# ------------------------------------------------------- codes and degradations
def test_unknown_type_code_is_dropped(tmp_path):
    head = "\n".join(["1 1", "6 1 1 0 1", "1 1 1 1 0 0 0", "3 0 0 1", "1", "0", "0",
                      "0 1 1e-6 1 1 0 0", "0 1 1e-6 1 1 0 0", "0 1 1e-6 1 1 0 0",
                      "0 1e6 938272088.16 1 1e9 0"])
    deck = tmp_path / "ImpactZ.in"
    deck.write_text(head + "\n0.5 1 1 77 1.0 /\n0 0 0 -99 /\n")
    lat, rep = Reader().read(deck)
    (el,) = [p.element for p in lat.flatten()]
    assert isinstance(el, Drift) and el.length == pytest.approx(0.5)
    assert "UNSUPPORTED_IMPACTZ_TYPE" in rep.codes()
    assert el.native["impactz"] == {"type": 77, "class": "type77", "nseg": 1, "mapstp": 1,
                                    "values": [1.0], "line": 12, "passthrough": True}


def test_collimator_and_kicker_from_negative_codes(tmp_path):
    head = "\n".join(["1 1", "6 1 1 0 1", "1 1 1 1 0 0 0", "3 0 0 1", "1", "0", "0",
                      "0 1 1e-6 1 1 0 0", "0 1 1e-6 1 1 0 0", "0 1 1e-6 1 1 0 0",
                      "0 1e6 938272088.16 1 1e9 0"])
    deck = tmp_path / "ImpactZ.in"
    deck.write_text(head + "\n0 0 0 -13 0 -0.01 0.02 -0.03 0.04 /\n"
                           "0 0 0 -21 0 0 1e-3 0 -2e-3 0 0 /\n0 0 0 -99 /\n")
    lat, rep = Reader().read(deck)
    col, kick = (p.element for p in lat.flatten())
    assert col.kind == "Collimator" and col.aperture.x_limits == pytest.approx((-0.01, 0.02))
    assert col.aperture.y_limits == pytest.approx((-0.03, 0.04))
    assert kick.kind == "Kicker" and kick.hkick == pytest.approx(1e-3)
    assert kick.vkick == pytest.approx(-2e-3)
    assert rep.counts.get("LOSSY", 0) == 0


def test_species_matching():
    assert species_from_header(Header(mass_eV=species("proton").mass_eV, charge=1.0)).name \
        == "proton"
    assert species_from_header(Header(mass_eV=species("h-").mass_eV, charge=-1.0)).name == "h-"
    odd = species_from_header(Header(mass_eV=511005.0, charge=-1.0))
    assert odd.charge == -1 and odd.mass_eV == pytest.approx(511005.0)


# -------------------------------------------------------------- reader options
def test_rfdata_is_loaded_when_present(tmp_path):
    head = "\n".join(["1 1", "6 1 1 0 1", "1 1 1 1 0 0 0", "3 0 0 1", "1", "0", "0",
                      "0 1 1e-6 1 1 0 0", "0 1 1e-6 1 1 0 0", "0 1 1e-6 1 1 0 0",
                      "0 1e6 938272088.16 1 1e9 0"])
    (tmp_path / "rfdata7.in").write_text("0 0 0 0\n0.5 1e6 0 0\n1.0 0 0 0\n")
    deck = tmp_path / "ImpactZ.in"
    deck.write_text(head + "\n1.0 1 20 104 1 1e9 -30 7 0.02 /\n0 0 0 -99 /\n")
    lat, rep = Reader().read(deck)
    fm = lat.flatten()[0].element
    assert fm.kind == "FieldMap"
    assert fm.meta["impactz_rfdata_columns"] == 4
    assert len(fm.meta["impactz_rfdata"]) == 3
    assert "IMPACTZ_RFDATA_MISSING" not in rep.codes()
    lat2, _ = Reader().read(deck, load_rfdata=False)
    assert "impactz_rfdata" not in lat2.flatten()[0].element.meta


def test_type_names_cover_every_code_the_writer_emits(tmp_path):
    from lattix.formats.impactz.writer import Writer as W

    ref = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6,
                            rf_frequency_Hz=1e9)
    from lattix.ir.lattice import Lattice

    lat = Lattice.from_sequence("s", [Drift(name="d", length=1.0)], ref)
    out = tmp_path / "ImpactZ.in"
    W().write(lat, out)
    for c in parse_deck(out.read_text())[1]:
        assert c.itype in TYPE_NAMES


def test_writer_reader_round_trip_of_a_read_deck(tmp_path):
    lat, _ = read("Example3")
    out = tmp_path / "ImpactZ.in"
    Writer().write(lat, out)
    back, _ = Reader().read(out)
    assert [p.element.kind for p in back.flatten()] == [p.element.kind for p in lat.flatten()]
    assert back.total_length == pytest.approx(lat.total_length)
