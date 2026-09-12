"""TraceWin writer: card formatting, FREQ / SET_SYNC_PHASE state, field-map paths, RULES coverage,
golden snapshots, write∘read idempotence (I-13), round-trip invariants and the byte comparison with
HELIX's own writer (style of HELIX tests/io/test_tracewin_writer.py)."""

from __future__ import annotations

import math
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import pytest

from lattix.formats.base import check_rules_coverage
from lattix.formats.tracewin import Reader, Writer, read, render, write
from lattix.ir import (
    ALL_KINDS,
    RFP,
    ApertureP,
    Bend,
    BendP,
    BodyShiftP,
    Collimator,
    Directive,
    Drift,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    Lattice,
    Line,
    LineItem,
    MagneticMultipoleP,
    Marker,
    Quadrupole,
    ReferenceChange,
    ReferenceParticle,
    RFCavity,
    RFQCell,
    Sextupole,
    Solenoid,
    SolenoidP,
    Superposition,
    propagate,
    species,
)
from lattix.ir.units import C_LIGHT, wrap_rad

PUBLIC = Path(__file__).resolve().parents[1] / "data" / "public"
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden" / "tracewin"
GOLDEN_DECKS = ["fodo_cell", "bend_line", "dtl_section", "solenoid_channel", "mebt_line"]
PUBLIC_DECKS = sorted(p for p in PUBLIC.rglob("*.dat"))

PROTON = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)
HMINUS = ReferenceParticle(species=species("h-"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)


def _lat(*elements, ref=PROTON):
    return Lattice.from_sequence("l", list(elements), ref)


def _lines(lat, out_dir=None, **options) -> list[str]:
    text, _ = render(lat, out_dir, **options)
    # the reference-particle tag is a header comment, not a card
    return [ln for ln in text.splitlines() if not ln.startswith("; lattix: reference")]


def _cards(text: str) -> list[str]:
    """Deck lines without the reference-particle header tag."""
    return [ln for ln in text.splitlines() if not ln.startswith("; lattix: reference")]


def _normalise(text: str) -> str:
    """Golden-file normaliser: absolute FIELD_MAP_PATH directories are machine specific."""
    return re.sub(r"^FIELD_MAP_PATH .*$", "FIELD_MAP_PATH <ABS>", text, flags=re.M)


# ── RULES coverage ────────────────────────────────────────────────────────────────────────────
def test_rules_cover_every_kind():
    w = Writer()
    assert check_rules_coverage(w) == set()
    assert set(w.RULES) == set(ALL_KINDS)
    assert all(callable(r) for r in w.RULES.values())
    assert w.format == "tracewin" and Reader().format == "tracewin"


# ── card formatting ─────────────────────────────────────────────────────────────────────────
def test_drift_forms():
    lat = _lat(
        Drift(name="a", length=0.05, aperture=ApertureP.circle(0.03)),
        Drift(name="b", length=0.05, aperture=ApertureP.rect(0.03, 0.02)),
        Drift(
            name="c",
            length=0.1,
            aperture=ApertureP.rect(0.02, 0.01),
            shift=BodyShiftP(x_offset=0.5e-3, y_offset=-0.3e-3),
        ),
        Drift(name="d", length=0.1, aperture=ApertureP.circle(0.02), shift=BodyShiftP(y_offset=1e-3)),
        Drift(name="e", length=1e-23),
    )
    assert _lines(lat)[:-1] == [
        "a: DRIFT 50 30",
        "b: DRIFT 50 30 20",
        "c: DRIFT 100 20 10 0.5 -0.3",
        "d: DRIFT 100 20 0 0 1",
        "e: DRIFT 1e-20 0",
    ]


def test_quad_sextupole_and_elision():
    q = Quadrupole(
        name="q",
        length=0.05,
        aperture=ApertureP.circle(0.02),
        multipole=MagneticMultipoleP(Bn={1: 5.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}, tilt={1: math.radians(30)}),
        native={"tracewin": {"gfr": 7.0}},
    )
    q2 = Quadrupole(name="q2", length=0.05, multipole=MagneticMultipoleP(Bn={1: -5.0}))
    s = Sextupole(
        name="s", length=0.1, aperture=ApertureP.circle(0.02), multipole=MagneticMultipoleP(Bn={2: 3.0})
    )
    lines, rep = render(_lat(q, q2, s))
    assert _cards(lines)[:-1] == [
        "q: QUAD 50 5 20 30 1 2 3 4 7",
        "q2: QUAD 50 -5 0",
        "s: QUAD 100 0 20 0 3",
    ]
    assert rep.codes() == {"THICK_MULTIPOLE_AS_QUAD_CARD": 1}


def test_bend_from_ir_emits_edges_and_field_index():
    G = 0.3
    b = Bend(
        name="b",
        length=1.0,
        aperture=ApertureP.circle(0.025),
        bend=BendP(angle=0.1, e1=0.02, e2=0.03, edge_int1=0.5, hgap=0.02),
        multipole=MagneticMultipoleP(Bn={1: G}),
    )
    lines = _lines(_lat(b))
    rho_mm = 1.0 / 0.1 * 1e3
    N = -(G / PROTON.brho_signed) * 10.0**2
    assert lines[0] == f"EDGE {math.degrees(0.02):.15g} {rho_mm:.15g} 40 0.5 2.8 25"
    assert lines[1] == f"b: BEND {math.degrees(0.1):.15g} {rho_mm:.15g} {N:.15g} 25"
    assert lines[2] == f"EDGE {math.degrees(0.03):.15g} {rho_mm:.15g} 40 0.5 2.8 25"
    # a plain sector bend without pole faces gets no EDGE cards
    plain = Bend(name="p", length=0.5, bend=BendP(angle=-0.2))
    assert _lines(_lat(plain))[0] == f"p: BEND {math.degrees(-0.2):.15g} 2500"


@pytest.mark.parametrize("tilt,hv,sign", [(math.pi / 2, "1", 1), (-math.pi / 2, "1", -1), (0.0, None, 1)])
def test_bend_vertical_plane(tilt, hv, sign):
    b = Bend(name="b", length=0.5, bend=BendP(angle=0.2, tilt_ref=tilt))
    line = _lines(_lat(b))[0]
    theta = f"{sign * math.degrees(0.2):.15g}"
    assert line == (f"b: BEND {theta} 2500 0 0 {hv}" if hv else f"b: BEND {theta} 2500")


def test_bend_field_index_round_trip_through_local_rigidity(tmp_path):
    p = tmp_path / "b.dat"
    p.write_text(
        "FREQ 162.5\nSET_SYNC_PHASE\nGAP 3e6 0 20\nEDGE 0 1000 20\nBEND 30 1000 0.35 20\n"
        "EDGE 0 1000 20\nEND\n"
    )
    lat, _ = read(p, species="h-")
    lines = _lines(lat, tmp_path)
    # the BEND aperture is the Bend's aperture, so both EDGE cards carry R=20 (and hence K1/K2 explicitly)
    assert lines == [
        "FREQ 162.5",
        "SET_SYNC_PHASE",
        "GAP 3000000 0 20",
        "EDGE 0 1000 20 0.45 2.8 20",
        "BEND 30 1000 0.35 20",
        "EDGE 0 1000 20 0.45 2.8 20",
        "END",
    ]


def test_solenoid_and_rfcavity_freq_state():
    sol = Solenoid(name="s", length=0.2, aperture=ApertureP.circle(0.02), solenoid=SolenoidP(Bsol_T=0.5))
    g1 = RFCavity(name="g1", rf=RFP(voltage_V=1.5e6, phase_rad=math.radians(-30), frequency_Hz=352.21e6))
    g2 = RFCavity(name="g2", rf=RFP(voltage_V=1.5e6, phase_rad=math.radians(-30), frequency_Hz=352.21e6))
    g3 = RFCavity(
        name="g3",
        aperture=ApertureP.circle(0.015),
        rf=RFP(voltage_V=2.3e6, phase_rad=-0.1, frequency_Hz=704.42e6),
        native={"tracewin": {"p_flag": 1}},
    )
    lines = _lines(_lat(sol, g1, g2, g3))
    assert lines == [
        "s: SOLENOID 200 0.5 20",
        "FREQ 352.21",
        "SET_SYNC_PHASE",
        "g1: GAP 1500000 -30 0",
        "SET_SYNC_PHASE",
        "g2: GAP 1500000 -30 0",
        "FREQ 704.42",
        "SET_SYNC_PHASE",
        f"g3: GAP 2300000 {math.degrees(wrap_rad(-0.1)):.15g} 15 1",   # the writer wraps before converting
        "END",
    ]
    assert lines.count("FREQ 352.21") == 1


def test_header_frequency_option():
    lat = _lat(Drift(name="d", length=0.1), RFCavity(name="g", rf=RFP(voltage_V=1e6, frequency_Hz=352.21e6)))
    lines = _lines(lat, frequency_Hz=352.21e6)
    assert lines[0] == "FREQ 352.21" and lines.count("FREQ 352.21") == 1


def test_rf_phase_species_and_sync_forms():
    raw = RFCavity(
        name="r",
        rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30), frequency_Hz=162.5e6, phase_is_sync=False),
    )
    syn = RFCavity(
        name="s", rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30), frequency_Hz=162.5e6, phase_is_sync=True)
    )
    assert _lines(_lat(raw, syn, ref=HMINUS))[1:-1] == [
        "r: GAP 1000000 150 0",
        "SET_SYNC_PHASE",
        "s: GAP 1000000 -30 0",
    ]
    assert _lines(_lat(raw, syn, ref=PROTON))[1:-1] == [
        "r: GAP 1000000 -30 0",
        "SET_SYNC_PHASE",
        "s: GAP 1000000 -30 0",
    ]


def test_thick_cavity_becomes_drift_gap_drift():
    c = RFCavity(
        name="c",
        length=0.3,
        aperture=ApertureP.circle(0.02),
        rf=RFP(gradient_V_per_m=2e6, phase_rad=0.0, frequency_Hz=162.5e6),
    )
    text, rep = render(_lat(c))
    assert _cards(text)[1:-1] == ["DRIFT 150 20", "SET_SYNC_PHASE", "c: GAP 600000 0 20", "DRIFT 150 20"]
    assert rep.codes() == {"THICK_CAVITY_AS_GAP": 1}


def test_kicker_crossed_back_to_integrated_fields():
    brho = PROTON.brho_signed
    k = Kicker(name="k", hkick=-0.002 / brho, vkick=0.001 / brho, aperture=ApertureP.circle(0.02))
    erho = PROTON.beta * C_LIGHT * brho
    e = Kicker(name="e", hkick=500 / erho, vkick=-200 / erho, electric=True)
    lines = _lines(_lat(k, e))
    assert lines[0] == "k: THIN_STEERING 0.001 -0.002 20"
    assert lines[1] == "e: THIN_STEERING 500 -200 0 1"
    # after a cavity the local rigidity is used
    g = RFCavity(name="g", rf=RFP(voltage_V=1e6, phase_rad=0.0, frequency_Hz=162.5e6))
    ref = PROTON.advanced(dE_eV=1e6)
    k2 = Kicker(name="k2", hkick=0.003 / ref.brho_signed)
    assert _lines(_lat(g, k2))[-2] == "k2: THIN_STEERING 0 0.003"


def test_collimator_types():
    c0 = Collimator(name="c0", aperture=ApertureP.rect(0.015, 0.010))
    c1 = Collimator(name="c1", aperture=ApertureP.circle(0.008))
    c5 = Collimator(name="c5", aperture=ApertureP.rect(0.0105, 0.009), native={"tracewin": {"ap_type": 5}})
    assert _lines(_lat(c0, c1, c5))[:-1] == [
        "c0: APERTURE 15 10 0",
        "c1: APERTURE 8 8 1",
        "c5: APERTURE 10.5 9 5",
    ]


def test_markers_instruments_and_foreign_family():
    m = Marker(name="m")
    bpm = Instrument(
        name="BPM_0001",
        family="BPM",
        native={"tracewin": {"card": "BPM", "raw": "BPM", "args": [], "label_only": True}},
    )
    dp = Instrument(
        name="D01BPM",
        family="DIAG_POSITION",
        params={"diag": 12},
        native={
            "tracewin": {
                "card": "DIAG_POSITION",
                "raw": "DIAG_POSITION",
                "args": ["12", "1e50", "0.5"],
                "label_only": False,
            }
        },
    )
    mon = Instrument(name="mon", family="MONITOR")
    prof = Instrument(name="prof", family="PROFILE")
    text, rep = render(_lat(m, bpm, dp, mon, prof))
    assert _cards(text)[:-1] == [
        "m: MARKER",
        "BPM :",
        "D01BPM: DIAG_POSITION 12 1e50 0.5",
        "BPM :",
        "prof: MARKER",
    ]
    assert rep.codes() == {"INSTRUMENT_AS_MARKER": 2}


def test_foil_reference_change_freq_and_directives():
    els = [
        Foil(
            name="F1", material="C", thickness_kg_per_m2=1e-3, native={"tracewin": {"straggling": "landau"}}
        ),
        ReferenceChange(name="r1", energy_eV=2.5e6),
        ReferenceChange(name="r2", dE_ref_eV=-0.1e6),
        ReferenceChange(
            name="r3", energy_eV=3e6, native={"tracewin": {"card": "SET_BEAM_ENERGY", "args": ["1", "3.0"]}}
        ),
        Freq(name="f", frequency_Hz=162.5e6),
        Directive(name="t", card="TITLE", args=["My", "deck"], role="title"),
        Directive(name="lg", card="@LG", args=["dc_kernel=fast"], role="tracking"),
        Directive(name="sc", card="HELIX_SC_GRID", args=["20"], role="tracking"),
        Directive(name="lat", card="LATTICE", args=["5", "0"], role="period_start"),
        Directive(name="e", card="ERROR_QUAD_NCPL_STAT", args=["1", "2"], role="error"),
    ]
    text, rep = render(_lat(*els))
    assert _cards(text) == [
        "; HELIX_FOIL F1 C 100 landau",
        "SET_BEAM_ENERGY 1 2.5",
        "SET_BEAM_E0_P0 1 -0.1 0 1 0",
        "SET_BEAM_ENERGY 1 3.0",
        "FREQ 162.5",
        "; TITLE My deck",
        ";@LG dc_kernel=fast",
        "; HELIX_SC_GRID 20",
        "LATTICE 5 0",
        "ERROR_QUAD_NCPL_STAT 1 2",
        "END",
    ]
    assert "TITLE My deck" not in [ln for ln in text.splitlines() if not ln.startswith(";")]
    assert rep.codes() == {"FOIL_AS_COMMENT": 1}


def test_labels_and_name_tags(tmp_path):
    lat = _lat(
        Drift(name="DRIFT_0001", length=0.1),
        Drift(name="qf.1", length=0.1),
        Drift(name="my drift", length=0.1),
        Drift(name="Q1_2", length=0.1, provenance=None),
    )
    lines = _lines(lat)
    assert lines[:-1] == [
        "DRIFT 100 0",
        "qf.1: DRIFT 100 0",
        '; lattix: name="my drift" type="Drift"',
        "DRIFT 100 0",
        "Q1_2: DRIFT 100 0",
    ]
    out = tmp_path / "o.dat"
    write(lat, out)
    back, _ = read(out)
    assert [p.name for p in back.flatten()] == ["DRIFT_0001", "qf.1", "my drift", "Q1_2"]


def test_duplicate_deck_label_written_with_original_name(tmp_path):
    p = tmp_path / "d.dat"
    p.write_text("D01T: THIN_STEERING 1e-5 0 20 0\nD01T: THIN_STEERING 2e-5 0 20 0\nEND\n")
    lat, _ = read(p)
    assert [ln.split(":")[0] for ln in _lines(lat, tmp_path)[:-1]] == ["D01T", "D01T"]


def test_ncells_and_rfq_from_params(tmp_path):
    p = tmp_path / "n.dat"
    p.write_text(
        "FREQ 162.5\nNCELLS 1 4 0.1 5.4e6 -30 15 1 0 0 0 0\n"
        "NCELLS 0 2 0.2 1e6 -20 15 0 0 0 0 0 0.5 0.8 0.1 0.01\n"
        "RFQ_CELL 80000 3.5 1.2 1.5 12.5 -30 2 1 0.5\nRFQ_CELL 80000 3.5 1.2 1.5 12.5 -30 0 0 0\nEND\n"
    )
    lat, _ = read(p, species="proton")
    lines = _lines(lat, tmp_path)
    assert lines == [
        "FREQ 162.5",
        "NCELLS 1 4 0.1 5400000 -30 15 1",
        "NCELLS 0 2 0.2 1000000 -20 15 0 0 0 0 0 0.5 0.8 0.1 0.01",
        "RFQ_CELL 80000 3.5 1.2 1.5 12.5 -30 2 1 0.5",
        "RFQ_CELL 80000 3.5 1.2 1.5 12.5 -30 0",
        "END",
    ]
    cell = RFQCell(name="x", length=0.01, params={"voltage": 1.0})
    _, rep = render(_lat(cell))
    assert "RFQ_PARAMS_MISSING" in rep.codes()


# ── field maps ───────────────────────────────────────────────────────────────────────────────
def _fm(name, map_dir, base="cav", **kw):
    return FieldMap(
        name=name,
        length=0.2,
        geom=100,
        files=[base],
        aperture=ApertureP.circle(0.02),
        rf=RFP(frequency_Hz=352.2e6, phase_rad=math.radians(153.171), phase_is_sync=False),
        meta={"field_map_dir": str(map_dir)},
        **kw,
    )


def test_field_map_path_relative_to_output_dir(tmp_path):
    maps = tmp_path / "maps"
    maps.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    lat = _lat(
        _fm("a", maps),
        _fm("b", maps, base="other"),
        _fm("c", tmp_path / "maps2"),
        _fm("d", out, base="local"),
    )
    lines = _lines(lat, out)
    assert lines == [
        "FREQ 352.2",
        "FIELD_MAP_PATH ../maps",
        "a: FIELD_MAP 100 200 153.171 20 1 1 0 1 cav",
        "b: FIELD_MAP 100 200 153.171 20 1 1 0 1 other",
        "FIELD_MAP_PATH ../maps2",
        "c: FIELD_MAP 100 200 153.171 20 1 1 0 1 cav",
        "FIELD_MAP_PATH .",
        "d: FIELD_MAP 100 200 153.171 20 1 1 0 1 local",
        "END",
    ]
    # a map next to the output deck needs no FIELD_MAP_PATH at all
    assert (
        _lines(_lat(_fm("d", out, base="local")), out)[1] == "d: FIELD_MAP 100 200 153.171 20 1 1 0 1 local"
    )


def test_field_map_absolute_fallback_warns_once(tmp_path):
    far = "/lattix_nonexistent_far_away/maps"
    lat = _lat(_fm("a", far), _fm("b", far))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lines = _lines(lat, tmp_path)
    assert lines[1] == f"FIELD_MAP_PATH {far}" and lines.count(f"FIELD_MAP_PATH {far}") == 1
    assert len([x for x in w if "not relocatable" in str(x.message)]) == 1
    assert any("not relocatable" in s for s in lat.warnings)


def test_field_map_quoting_and_superposition(tmp_path):
    maps = tmp_path / "my maps"
    maps.mkdir()
    (maps / "cav.edz").write_text("1\n0 1\n0\n1\n")
    p = tmp_path / "s.dat"
    p.write_text(
        'FIELD_MAP_PATH "my maps"\nFREQ 352.2\nSUPERPOSE_MAP 0\n'
        "FIELD_MAP 100 415.16 153.171 30 0 1.55425 0 0 cav\n"
        "SUPERPOSE_MAP 680 0 0\nFIELD_MAP 100 415.16 156.892 30 0 1.55425 0 0 cav\nDRIFT 10 5\nEND\n"
    )
    lat, rep = read(p, species="proton")
    assert rep.ok
    assert isinstance(lat.flatten()[1].element, Superposition)
    out = tmp_path / "sub" / "o.dat"
    write(lat, out)
    lines = _cards(out.read_text())
    assert lines == [
        "FREQ 352.2",
        'FIELD_MAP_PATH "../my maps"',
        "SUPERPOSE_MAP 0",
        "FIELD_MAP 100 415.16 153.171 30 0 1.55425 0 0 cav",
        "SUPERPOSE_MAP 680 0 0",
        "FIELD_MAP 100 415.16 156.892 30 0 1.55425 0 0 cav",
        "DRIFT 10 5",
        "END",
    ]
    back, _ = read(out, species="proton")
    assert back.total_length == pytest.approx(lat.total_length)
    assert [q.element.kind for q in back.flatten()] == [q.element.kind for q in lat.flatten()]


def test_superposition_from_ir_children(tmp_path):
    lat = Lattice(name="l", reference=PROTON)
    a = lat.add_element(_fm("a", tmp_path))
    b = lat.add_element(_fm("b", tmp_path))
    s = lat.add_element(Superposition(name="s", length=0.9, children=[(0.0, a), (0.7, b)]))
    lat.lines["l"] = Line(name="l", items=[LineItem(ref=s)])
    lat.use = "l"
    lines = _lines(lat, tmp_path)
    assert lines[1:3] == ["SUPERPOSE_MAP 0", "a: FIELD_MAP 100 200 153.171 20 1 1 0 1 cav"]
    assert lines[3:5] == ["SUPERPOSE_MAP 700", "b: FIELD_MAP 100 200 153.171 20 1 1 0 1 cav"]


def test_negative_zero_and_precision():
    q = Quadrupole(name="q", length=0.123456789012, multipole=MagneticMultipoleP(Bn={1: -0.0}))
    assert _lines(_lat(q))[0] == "q: QUAD 123.456789012 0 0"


def test_write_creates_parent_and_latin1(tmp_path):
    lat = _lat(Directive(name="t", card="TITLE", args=["café"], role="title"), Drift(name="d", length=0.1))
    out = tmp_path / "deep" / "dir" / "o.dat"
    rep = write(lat, out)
    assert b"\n; TITLE caf\xe9\nd: DRIFT" in out.read_bytes() and rep.target_file == str(out)
    assert rep.target_format == "tracewin"


# ── golden snapshots, idempotence, invariants ──────────────────────────────────────────────────
@pytest.mark.parametrize("stem", GOLDEN_DECKS)
def test_golden_snapshot(tmp_path, stem):
    lat, _ = read(PUBLIC / "helix" / f"{stem}.dat", species="proton")
    out = tmp_path / f"{stem}.dat"
    write(lat, out)
    got = _normalise(out.read_text(encoding="latin-1"))
    golden = GOLDEN_DIR / f"{stem}.dat"
    if os.environ.get("LATTIX_UPDATE_GOLDENS"):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(got, encoding="latin-1")
    assert golden.exists(), f"missing golden {golden}; run with LATTIX_UPDATE_GOLDENS=1"
    assert got == golden.read_text(encoding="latin-1")


@pytest.mark.parametrize("deck", PUBLIC_DECKS, ids=[p.relative_to(PUBLIC).as_posix() for p in PUBLIC_DECKS])
def test_write_read_fixed_point_and_invariants(tmp_path, deck):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat1, _ = read(deck, species="proton")
        out1 = tmp_path / "pass1.dat"
        write(lat1, out1)
        lat2, _ = read(out1, species="proton")
        out2 = tmp_path / "pass2.dat"
        write(lat2, out2)
    assert out1.read_bytes() == out2.read_bytes()  # I-13 fixed point
    p1, p2 = propagate(lat1, warnings=[]), propagate(lat2, warnings=[])
    assert lat2.total_length == pytest.approx(lat1.total_length, rel=1e-12)  # I-1
    assert [p.element.kind for p in p2] == [p.element.kind for p in p1]
    assert [p.length for p in p2] == pytest.approx([p.length for p in p1], rel=1e-12)
    assert p2[-1].ref_out.kinetic_energy_eV == pytest.approx(
        p1[-1].ref_out.kinetic_energy_eV, rel=1e-12
    )  # I-6/I-18
    assert lat2.reference.rf_frequency_Hz == pytest.approx(lat1.reference.rf_frequency_Hz)


def test_h_minus_round_trip_keeps_raw_phases(tmp_path):
    src = PUBLIC / "lightwin" / "example.dat"
    lat, _ = read(src, species="h-")
    out = tmp_path / "o.dat"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write(lat, out)
    orig = [ln.split()[3] for ln in src.read_text().splitlines() if ln.startswith("FIELD_MAP ")]
    got = [ln.split()[3] for ln in out.read_text().splitlines() if ln.startswith("FIELD_MAP ")]
    assert [float(x) for x in got] == pytest.approx([float(x) for x in orig], abs=1e-9)


# ── byte comparison against HELIX's writer ───────────────────────────────────────────────────
def _helix():
    try:
        from lattix.oracles.helix import _import_helix

        root = _import_helix()
        import linac_gen.io.tracewin_parser  # noqa: F401  (HELIX's own writer needs scipy: skip, not error)
        import linac_gen.io.tracewin_writer  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"HELIX not importable: {exc}")
    deck = root / "examples" / "pipii" / "mebt" / "mebt.dat"
    if not deck.is_file() or not (root / "Fields").is_dir():
        pytest.skip("HELIX mebt.dat / Fields not available")
    return root, deck


# Every differing line must belong to one of these explained classes (see the module report).
_DIFF_CLASSES = {
    "drift_ry0_elided": lambda ln: ln.startswith("DRIFT "),
    "hardware_marker_verbatim": lambda ln: (
        ln in ("MARKER", "BPM") or ln.endswith(" :") or ln.startswith("CHOPPER ")
    ),
    "fieldmap_path_and_sync_phase_card": lambda ln: ln.startswith(("FIELD_MAP", "SET_SYNC_PHASE")),
    "steerer_aperture_kept": lambda ln: ln.startswith("THIN_STEERING "),
    "title_comment": lambda ln: ln.startswith("; TITLE"),
    "edge_from_e1_e2": lambda ln: ln.startswith("EDGE "),
    "gap_ttf_folded": lambda ln: ln.startswith("GAP "),
    "label_style": lambda ln: re.match(r"^[A-Za-z][^\s:]*: ", ln) is not None,
}


def _canon(lines: list[str], *, helix: bool) -> list[str]:
    """Map both writers' output onto their common information content."""
    out = []
    for ln in lines:
        t = ln.split()
        if not t or ln.startswith("; TITLE"):
            continue
        if helix:
            if t[0] == "DRIFT" and len(t) == 4 and t[3] == "0":
                t = t[:3]  # HELIX keeps the explicit Ry=0 slot
            if t[0] == "FIELD_MAP":
                t = t[:9] + [os.path.basename(t[9])]  # absolute path in the card, p_flag=1 for sync
            if t[0] == "BPM":
                t = ["MARKER"]
        else:
            if t[0] == "FIELD_MAP_PATH":
                continue
            if t[0] == "FIELD_MAP":
                t = t[:10]
            if (len(t) == 2 and t[1] == ":") or t[0] == "CHOPPER":
                t = ["MARKER"]
        if t[0] == "THIN_STEERING":
            t = t[:3]
        out.append(" ".join(t))
    return out


@pytest.mark.oracle_helix
def test_byte_compare_with_helix_writer_on_mebt(tmp_path):
    root, deck = _helix()
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hl, _meta = parse_tracewin(str(deck))
        write_tracewin(hl, str(tmp_path / "helix.dat"))
        lat, rep = read(deck, species="h-")
        write(lat, tmp_path / "lattix.dat")
    a = (tmp_path / "helix.dat").read_text(encoding="latin-1").splitlines()
    b = _cards((tmp_path / "lattix.dat").read_text(encoding="latin-1"))
    # multiset difference (positional diffs misalign identical neighbouring lines)
    only_helix = list((Counter(a) - Counter(b)).elements())
    only_lattix = list((Counter(b) - Counter(a)).elements())
    unexplained, classes = [], {}
    for sign, lines in (("-", only_helix), ("+", only_lattix)):
        for body in lines:
            cls = next((k for k, f in _DIFF_CLASSES.items() if f(body)), None)
            if cls is None:
                unexplained.append(sign + body)
            classes[cls] = classes.get(cls, 0) + 1
    print("HELIX-writer diff classes on mebt.dat:", classes)
    assert not unexplained, unexplained[:10]
    assert set(classes) <= set(_DIFF_CLASSES)
    assert _canon(a, helix=True) == _canon(b, helix=False)
    assert len(a) == len(b) - sum(1 for ln in b if ln.startswith("FIELD_MAP_PATH"))


def test_trailing_comments_survive_the_round_trip(tmp_path):
    """The writer puts a card's comment back on its line, so a deck that names its markers by comments
    reads, writes and re-reads to the same names and kinds, and the second write equals the first."""
    src = tmp_path / "named.dat"
    src.write_text("DRIFT 300 25.4\n"
                   "DRIFT 0.000 25.400 ; 4.898 HKV MONITOR\n"
                   "QF1:QUAD 200 5.0 25.4 ; 10.295 QF1 QUADRUPOLE K1=0.5\n"
                   "DRIFT 200 25.4 ; drift to the buncher\n"
                   "MARKER ; 12.5 DCH01 HKICKER\n"
                   "END\n")
    lat, _ = read(src, kinetic_energy_eV=2.1e6)
    text = render(lat)
    text = text[0] if isinstance(text, tuple) else text
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(";")]
    assert "DRIFT 0.000 25.400 ; 4.898 HKV MONITOR" in lines
    assert any(ln.startswith("QF1: QUAD") and ln.endswith("; 10.295 QF1 QUADRUPOLE K1=0.5") for ln in lines), lines
    assert any(ln.startswith("DRIFT") and ln.endswith("; drift to the buncher") for ln in lines)
    assert "MARKER ; 12.5 DCH01 HKICKER" in lines
    out = tmp_path / "again.dat"
    out.write_text(text)
    lat2, _ = read(out, kinetic_energy_eV=2.1e6)
    line1 = next(iter(lat.lines.values())).items
    line2 = next(iter(lat2.lines.values())).items
    assert [(lat.elements[i.ref].name, lat.elements[i.ref].kind) for i in line1] == \
           [(lat2.elements[i.ref].name, lat2.elements[i.ref].kind) for i in line2]
    again = render(lat2)
    assert (again[0] if isinstance(again, tuple) else again) == text
