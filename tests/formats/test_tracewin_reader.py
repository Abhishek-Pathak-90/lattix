"""TraceWin reader unit tests on inline decks (style of HELIX tests/io/test_tracewin_parser.py)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix.formats.tracewin import Reader, read
from lattix.ir import (
    Bend,
    Collimator,
    Directive,
    Drift,
    FieldMap,
    Foil,
    Freq,
    Instrument,
    Kicker,
    Marker,
    NCells,
    Quadrupole,
    ReferenceChange,
    ReferenceParticle,
    RFCavity,
    RFQCell,
    Solenoid,
    Superposition,
    propagate,
    species,
)
from lattix.ir.units import C_LIGHT

PUBLIC = Path(__file__).resolve().parents[1] / "data" / "public"
PROTON = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6)
HMINUS = ReferenceParticle(species=species("h-"), kinetic_energy_eV=2.1e6)


def _read(tmp_path, text: str, **kw):
    p = tmp_path / "deck.dat"
    p.write_text(text)
    kw.setdefault("kinetic_energy_eV", 2.1e6)          # a stated beam: these tests are about the cards
    return read(p, **kw)


def _els(lat):
    return [p.element for p in lat.flatten()]


def _first(lat, cls):
    return next(e for e in _els(lat) if isinstance(e, cls))


# -- drifts / quads / solenoids ------------------------------------------------------------------
@pytest.mark.parametrize(
    "line,length,ap,shift",
    [
        ("DRIFT 50 30", 0.05, ("ELLIPTICAL", 0.03, 0.03), None),
        ("DRIFT 50 30 20", 0.05, ("RECTANGULAR", 0.03, 0.02), None),
        ("DRIFT 100 20 10 0.5 -0.3", 0.1, ("RECTANGULAR", 0.02, 0.01), (0.5e-3, -0.3e-3)),
        ("DRIFT 150 100 0 0 0", 0.15, ("ELLIPTICAL", 0.1, 0.1), None),
        ("DRIFT 1e-20 0", 1e-23, None, None),
    ],
)
def test_drift_forms(tmp_path, line, length, ap, shift):
    lat, rep = _read(tmp_path, f"{line}\nEND\n")
    d = _first(lat, Drift)
    assert d.length == pytest.approx(length, rel=1e-12)
    if ap is None:
        assert d.aperture is None
    else:
        assert d.aperture.shape == ap[0]
        assert d.aperture.half_x == pytest.approx(ap[1])
        assert d.aperture.half_y == pytest.approx(ap[2])
    if shift is None:
        assert d.shift is None
    else:
        assert (d.shift.x_offset, d.shift.y_offset) == pytest.approx(shift)
    assert rep.ok


def test_quad_all_positionals(tmp_path):
    lat, _ = _read(tmp_path, "QUAD 50 5 20 30 1 2 3 4 7\nEND\n")
    q = _first(lat, Quadrupole)
    assert q.length == pytest.approx(0.05)
    assert q.gradient == 5.0
    assert q.multipole.Bn == {1: 5.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0}
    assert q.skew_rad == pytest.approx(math.radians(30))
    assert q.native["tracewin"]["gfr"] == 7.0
    assert q.aperture.half_x == pytest.approx(0.02)
    assert q.name == "QUAD_0001"
    assert q.provenance.line == 1 and q.provenance.original_type == "QUAD"


def test_solenoid_and_extra_tokens(tmp_path):
    lat, rep = _read(tmp_path, "SOLENOID 200.0 0.5 20.0 10\nEND\n")
    s = _first(lat, Solenoid)
    assert s.length == pytest.approx(0.2) and s.solenoid.Bsol_T == 0.5
    assert rep.codes() == {"EXTRA_TOKENS_IGNORED": 1}
    assert rep.ok  # EQUIVALENT-class, not a problem


# -- RF: FREQ state, phases, species ----------------------------------------------------------
def test_freq_state_and_gap(tmp_path):
    lat, rep = _read(tmp_path, "FREQ 704.42\nGAP 1.5e6 -30.0 20.0\nFREQ 100\nGAP 1 0\nEND\n")
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Freq", "RFCavity", "Freq", "RFCavity"]
    g1, g2 = els[1], els[3]
    assert g1.rf.frequency_Hz == pytest.approx(704.42e6)
    assert g2.rf.frequency_Hz == pytest.approx(100e6)
    assert g1.rf.voltage_V == 1.5e6 and g1.rf.phase_rad == pytest.approx(math.radians(-30))
    assert g1.rf.phase_is_sync is False and g1.length == 0.0
    assert lat.reference.rf_frequency_Hz == pytest.approx(704.42e6)
    assert isinstance(els[0], Freq) and els[0].frequency_Hz == pytest.approx(704.42e6)
    assert rep.codes() == {"SPECIES_ASSUMED": 1}


def test_gap_before_freq_uses_default_and_option(tmp_path):
    lat, rep = _read(tmp_path, "GAP 1.5 -30.0 20.0\nEND\n", species="proton")
    assert _first(lat, RFCavity).rf.frequency_Hz == pytest.approx(352.21e6)
    assert "FREQ_ASSUMED" in rep.codes() and rep.ok
    assert lat.warnings and "352.21" in lat.warnings[0]
    lat2, _ = _read(tmp_path, "GAP 1.5 -30.0 20.0\nEND\n", species="proton", frequency_Hz=162.5e6)
    assert _first(lat2, RFCavity).rf.frequency_Hz == pytest.approx(162.5e6)
    assert lat2.reference.rf_frequency_Hz == pytest.approx(162.5e6)


def test_h_minus_raw_phase_gets_pi_shift(tmp_path):
    deck = "FREQ 162.5\nGAP 1e6 150 20\nEND\n"
    lat, rep = _read(tmp_path, deck, species="h-")
    g = _first(lat, RFCavity)
    assert g.rf.phase_rad == pytest.approx(math.radians(-30))
    assert "SPECIES_ASSUMED" not in rep.codes()
    # same deck read for a proton: raw phase taken as-is
    lat_p, rep_p = _read(tmp_path, deck)
    assert _first(lat_p, RFCavity).rf.phase_rad == pytest.approx(math.radians(150))
    assert rep_p.codes() == {"SPECIES_ASSUMED": 1}
    # the IR energy rule is species independent: H- at -30 deg gains V cos 30
    out = propagate(lat)[-1].ref_out.kinetic_energy_eV
    assert out == pytest.approx(2.1e6 + 1e6 * math.cos(math.radians(30)))


def test_set_sync_phase_pending_across_cards_and_one_shot(tmp_path):
    deck = "FREQ 162.5\nSET_SYNC_PHASE\nDRIFT 10 5\nQUAD 10 1 5\nGAP 1e6 -30 5\nGAP 1e6 -40 5\nEND\n"
    lat, rep = _read(tmp_path, deck, species="h-")
    els = _els(lat)
    assert isinstance(els[1], Directive) and els[1].role == "sync_phase" and els[1].card == "SET_SYNC_PHASE"
    gaps = [e for e in els if isinstance(e, RFCavity)]
    assert gaps[0].rf.phase_is_sync is True
    assert gaps[0].rf.phase_rad == pytest.approx(math.radians(-30))  # no π shift for a sync phase
    assert gaps[1].rf.phase_is_sync is False
    assert gaps[1].rf.phase_rad == pytest.approx(math.radians(-40 + 180))
    assert "SPECIES_ASSUMED" not in rep.codes()


def test_gap_p_flag_is_lossy(tmp_path):
    lat, rep = _read(tmp_path, "FREQ 352.21\nGAP 1.5e6 -30.0 20.0 1\nEND\n", species="proton")
    g = _first(lat, RFCavity)
    assert g.native["tracewin"]["p_flag"] == 1
    assert "GAP_ABSOLUTE_PHASE" in rep.codes() and not rep.ok


# -- labels -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "line",
    [
        "Q01: QUAD 85.3 -8.09 20 0 0 0 0 0",
        "Q01 : QUAD 85.3 -8.09 20",
        "Q01:QUAD 85.3 -8.09 20",
        "Q01 :QUAD 85.3 -8.09 20",
    ],
)
def test_label_grammar(tmp_path, line):
    lat, rep = _read(tmp_path, f"FREQ 804.96\n{line}\nEND\n")
    q = _first(lat, Quadrupole)
    assert q.name == "Q01" and q.provenance.original_name == "Q01"
    assert q.length == pytest.approx(0.0853) and q.gradient == pytest.approx(-8.09)
    assert rep.ok


def test_glued_colon_placeholder_label(tmp_path):
    lat, rep = _read(tmp_path, "FREQ 650\nskew: QUAD 200 0 22 0 0 0 0 0\nEND\n")
    q = _first(lat, Quadrupole)
    assert q.name == "skew" and q.gradient == 0.0 and q.length == pytest.approx(0.2)
    assert rep.ok


def test_duplicate_labels_are_uniquified_but_keep_original(tmp_path):
    lat, _ = _read(tmp_path, "D01T: THIN_STEERING 1e-5 0 20 0\nD01T: THIN_STEERING 2e-5 0 20 0\nEND\n")
    ks = [e for e in _els(lat) if isinstance(e, Kicker)]
    assert [k.name for k in ks] == ["D01T", "D01T_2"]
    assert all(k.provenance.original_name == "D01T" for k in ks)


@pytest.mark.parametrize(
    "line,family",
    [
        ("BPM :", "BPM"),
        ("BPM:", "BPM"),
        ("FastGV:", "FASTGV"),
        ("LB650 CM:", "LB650 CM"),
        ("Dump Entrance :", "DUMP ENTRANCE"),
        ("MEBTAbsorber :", "MEBTABSORBER"),
        ("ACCT :", "ACCT"),
    ],
)
def test_label_only_hardware_markers(tmp_path, line, family):
    lat, rep = _read(tmp_path, f"DRIFT 50 30\n{line}\nDRIFT 50 30\nEND\n")
    els = _els(lat)
    assert len(els) == 3
    m = els[1]
    assert isinstance(m, Instrument) and m.family == family and m.length == 0.0
    assert m.native["tracewin"]["label_only"] is True
    assert m.native["tracewin"]["raw"] == line.rstrip(" :").rstrip(":")
    assert rep.ok


def test_diag_position_label_and_sentinels(tmp_path):
    lat, rep = _read(
        tmp_path, "D01BPM: DIAG_POSITION 12 0.0035 -0.0007\nDIAG_POSITION 3 1e50 0.5 2\nBPM :\nEND\n"
    )
    a, b, c = _els(lat)
    assert isinstance(a, Instrument) and a.name == "D01BPM" and a.family == "DIAG_POSITION"
    assert a.params == {
        "diag": 12,
        "x_target_m": pytest.approx(3.5e-6),
        "y_target_m": pytest.approx(-0.7e-6),
        "accuracy_m": pytest.approx(1e-3),
    }
    assert b.params["x_target_m"] is None and b.params["y_target_m"] == pytest.approx(0.5e-3)
    assert b.params["accuracy_m"] == pytest.approx(2e-3)
    assert b.native["tracewin"]["args"] == ["3", "1e50", "0.5", "2"]
    assert isinstance(c, Instrument) and c.family == "BPM"


def test_diag_cards_and_markers(tmp_path):
    lat, rep = _read(
        tmp_path, "MARKER\nDIAG_SIZE 1\nDIAG_PHASE 1\nCHOPPER 2 00 8 0 1\nDIAG_WAIST 4 0 1\nEND\n"
    )
    els = _els(lat)
    assert isinstance(els[0], Marker)
    assert isinstance(els[1], Instrument) and els[1].family == "DIAG_SIZE" and els[1].params["diag"] == 1
    assert isinstance(els[2], Instrument) and els[2].family == "DIAG_PHASE"
    assert isinstance(els[3], Instrument) and els[3].family == "CHOPPER"
    assert els[3].native["tracewin"]["args"] == ["2", "00", "8", "0", "1"]
    assert isinstance(els[4], Directive) and els[4].role == "matching" and els[4].card == "DIAG_WAIST"
    assert all(e.length == 0.0 for e in els)
    assert rep.ok


# -- bends ------------------------------------------------------------------------------------
def test_bend_edge_cluster_into_one_bend(tmp_path):
    deck = (
        "FREQ 162.5\nEDGE 10 2000 30 0.5 3 25\nBEND 45 2000 0.5 25 0\nEDGE 12 2000 30 0.5 3 25\n"
        "DRIFT 10 5\nEND\n"
    )
    lat, rep = _read(tmp_path, deck)
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Freq", "Bend", "Drift"]
    b = els[1]
    assert b.length == pytest.approx(2.0 * math.pi / 4)
    assert b.bend.angle == pytest.approx(math.pi / 4)
    assert b.bend.e1 == pytest.approx(math.radians(10)) and b.bend.e2 == pytest.approx(math.radians(12))
    assert b.bend.hgap == pytest.approx(0.015)
    assert b.bend.edge_int1 == 0.5 and b.bend.edge_int2 == 0.5 and b.bend.fringe_k2 == 3.0
    assert b.bend.tilt_ref == 0.0 and b.aperture.half_x == pytest.approx(0.025)
    assert b.native["tracewin"]["has_edges"] is True
    assert b.native["tracewin"]["bend"]["field_index"] == 0.5
    # field index -> lab gradient through the local signed rigidity: G = -N Bρ / ρ²
    brho = PROTON.brho_signed
    assert b.gradient if hasattr(b, "gradient") else True
    assert b.multipole.Bn[1] == pytest.approx(-0.5 * brho / 2.0**2)
    assert rep.ok
    # H-: the signed rigidity flips (and differs by 2 m_e), so the lab gradient flips with it
    lat_h, _ = _read(tmp_path, deck, species="h-")
    assert _first(lat_h, Bend).multipole.Bn[1] == pytest.approx(-0.5 * HMINUS.brho_signed / 4.0)
    assert _first(lat_h, Bend).multipole.Bn[1] > 0


def test_bend_without_edges_and_vertical(tmp_path):
    lat, rep = _read(tmp_path, "BEND 45.0 2000.0 0.0 30.0 0\nBEND -10 1500 0 20 1\nEND\n")
    b1, b2 = _els(lat)
    assert b1.native["tracewin"]["has_edges"] is False and b1.bend.e1 == 0.0 and b1.bend.e2 == 0.0
    assert b1.multipole.Bn.get(1, 0.0) == 0.0
    assert b2.bend.tilt_ref == pytest.approx(math.pi / 2)
    assert b2.bend.angle == pytest.approx(math.radians(-10))
    assert b2.length == pytest.approx(1.5 * math.radians(10))
    assert rep.ok


def test_edge_bend_cluster_survives_error_card_between(tmp_path):
    """PIP-II BTL decks: EDGE / ERROR_BEND_NCPL_STAT / NAME:BEND / EDGE (found on the corpus)."""
    deck = (
        "EDGE 3.2818713 21384.55 50.000 1e-06 1e-06 25.4 0\nERROR_BEND_NCPL_STAT 1 0 0 0 0 0 0 0.1 0\n"
        "BA1012:BEND   6.5637426 21384.55 0.000 25.400 0\n"
        "EDGE 3.2818713 21384.55 50.000 1e-06 1e-06 25.4 0\nEND\n"
    )
    lat, rep = _read(tmp_path, deck)
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Directive", "Bend"]
    assert els[0].role == "error"
    b = els[1]
    assert (
        b.name == "BA1012" and b.bend.e1 == pytest.approx(math.radians(3.2818713)) and b.bend.e2 == b.bend.e1
    )
    assert b.bend.hgap == pytest.approx(0.025)
    assert rep.ok


def test_orphan_edge_kept_as_directive(tmp_path):
    lat, rep = _read(tmp_path, "EDGE 5 1000\nDRIFT 10 5\nBEND 10 1000\nEDGE 5 1000\nEDGE 6 1000\nEND\n")
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Directive", "Drift", "Bend", "Directive"]
    assert els[0].role == "edge" and els[0].args == ["5", "1000"]
    assert els[2].bend.e2 == pytest.approx(math.radians(5)) and els[2].bend.e1 == 0.0
    assert els[3].args == ["6", "1000"]
    assert rep.codes() == {"ORPHAN_EDGE": 2}


# -- kickers ----------------------------------------------------------------------------------
@pytest.mark.parametrize("card", ["THIN_STEERING", "STEERER"])
def test_steerer_magnetic_crossed_convention(tmp_path, card):
    lat, rep = _read(tmp_path, f"{card} 0.001 -0.002 20\nEND\n")
    k = _first(lat, Kicker)
    brho = PROTON.brho_signed
    assert k.electric is False and k.length == 0.0
    assert k.hkick == pytest.approx(-0.002 / brho)  # Δx' = ∫By·dl / Bρ
    assert k.vkick == pytest.approx(0.001 / brho)  # Δy' = ∫Bx·dl / Bρ
    assert k.aperture.half_x == pytest.approx(0.02)
    assert rep.ok
    lat_h, _ = _read(tmp_path, f"{card} 0.001 -0.002 20\nEND\n", species="h-")
    assert _first(lat_h, Kicker).hkick == pytest.approx(-0.002 / HMINUS.brho_signed)
    assert _first(lat_h, Kicker).hkick > 0


def test_steerer_electric_uses_electric_rigidity(tmp_path):
    lat, _ = _read(tmp_path, "THIN_STEERING 500 -200 20 1\nEND\n")
    k = _first(lat, Kicker)
    erho = PROTON.beta * C_LIGHT * PROTON.brho_signed
    assert k.electric is True
    assert k.hkick == pytest.approx(500 / erho) and k.vkick == pytest.approx(-200 / erho)


def test_kicker_uses_local_energy_after_gap(tmp_path):
    lat, _ = _read(tmp_path, "FREQ 162.5\nSET_SYNC_PHASE\nGAP 1e6 0 15\nTHIN_STEERING 0 0.001 15\nEND\n")
    ref = PROTON.advanced(dE_eV=1e6)
    assert _first(lat, Kicker).hkick == pytest.approx(0.001 / ref.brho_signed)


# -- apertures ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "line,shape,hx,hy,code",
    [
        ("APERTURE 15 10 0", "RECTANGULAR", 0.015, 0.010, None),
        ("APERTURE 8 0 1", "ELLIPTICAL", 0.008, 0.008, None),
        ("APERTURE 15 15 1", "ELLIPTICAL", 0.015, 0.015, None),
        ("APERTURE 6.8 15 0", "RECTANGULAR", 0.0068, 0.015, None),
        ("APERTURE 10.5 9 5", "RECTANGULAR", 0.0105, 0.009, "APERTURE_SHAPE"),
    ],
)
def test_aperture_card(tmp_path, line, shape, hx, hy, code):
    lat, rep = _read(tmp_path, f"{line}\nEND\n")
    c = _first(lat, Collimator)
    assert c.aperture.shape == shape
    assert c.aperture.half_x == pytest.approx(hx) and c.aperture.half_y == pytest.approx(hy)
    if code:
        assert rep.codes() == {code: 1} and c.native["tracewin"]["ap_type"] == 5
    else:
        assert rep.ok and "tracewin" not in c.native


# -- field maps --------------------------------------------------------------------------------
def _touch(*paths):
    for p in paths:
        Path(p).write_text("1\n0 1\n0\n1\n")


def test_field_map_resolution_and_files(tmp_path):
    maps = tmp_path / "maps"
    maps.mkdir()
    _touch(maps / "cav.edz", maps / "sol.bsz", maps / "q3.edx", maps / "q3.edy", maps / "q3.edz")
    deck = (
        "FIELD_MAP_PATH maps\nFREQ 352.2\nFIELD_MAP 100 415.16 153.171 30 0 1.55425 0 0 cav 0\n"
        "FIELD_MAP 10 200 0 25 1.2 0 0 0 sol\nFIELD_MAP 700 100 -25 18 1.0 1.1 0 0 q3\nEND\n"
    )
    lat, rep = _read(tmp_path, deck, species="proton")
    fms = [e for e in _els(lat) if isinstance(e, FieldMap)]
    assert len(fms) == 3 and rep.ok
    f = fms[0]
    assert f.geom == 100 and f.files == ["cav"] and f.length == pytest.approx(0.41516)
    assert f.ke == 1.55425 and f.kb == 0.0 and f.ki == 0.0 and f.ka == 0.0 and f.p_flag == 0
    assert f.rf.frequency_Hz == pytest.approx(352.2e6) and f.rf.phase_rad == pytest.approx(
        math.radians(153.171)
    )
    assert f.rf.phase_is_sync is False and f.rf.dE_ref_eV is None
    assert f.meta["field_map_dir"] == str(maps)
    assert f.meta["field_files"] == [str(maps / "cav.edz")] and f.meta["field_files_missing"] == []
    assert fms[1].meta["field_files"] == [str(maps / "sol.bsz")]
    assert sorted(Path(p).name for p in fms[2].meta["field_files"]) == ["q3.edx", "q3.edy", "q3.edz"]
    assert lat.meta["tracewin"]["field_map_path"] == str(maps)


def test_field_map_missing_files_kept_lossy(tmp_path):
    lat, rep = _read(tmp_path, "FIELD_MAP 7700 240 -90 20 0.068 0.068 0 0 QWR-2012-02\nEND\n", species="h-")
    f = _first(lat, FieldMap)
    assert f.length == pytest.approx(0.24)  # element kept (HELIX would drop it)
    assert len(f.meta["field_files_missing"]) == 6  # edx edy edz bdx bdy bdz
    assert rep.codes() == {"FM_FILES_MISSING": 1, "FREQ_ASSUMED": 1}
    assert not rep.ok


def test_field_map_sync_phase_and_negative_geom(tmp_path):
    deck = (
        "FREQ 162.5\nSET_SYNC_PHASE\nFIELD_MAP -100 240 -90 20 1 1 0 0 nofile\n"
        "FIELD_MAP 100 240 -90 20 1 1 0 0 nofile\nEND\n"
    )
    lat, rep = _read(tmp_path, deck, species="h-")
    a, b = [e for e in _els(lat) if isinstance(e, FieldMap)]
    assert (
        a.geom == -100 and a.rf.phase_is_sync is True and a.rf.phase_rad == pytest.approx(math.radians(-90))
    )
    assert b.rf.phase_is_sync is False and b.rf.phase_rad == pytest.approx(math.radians(90))
    assert set(rep.codes()) == {"FM_FILES_MISSING"}


def test_superpose_cluster_becomes_superposition(tmp_path):
    deck = (
        "FREQ 352.2\nDRIFT 10 30\nSUPERPOSE_MAP 0\nFIELD_MAP 100 415.16 153.171 30 0 1.55425 0 0 a\n"
        "SUPERPOSE_MAP 680\nFIELD_MAP 100 415.16 156.892 30 0 1.55425 0 0 b\nDRIFT 10 30\nEND\n"
    )
    lat, rep = _read(tmp_path, deck, species="proton")
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Freq", "Drift", "Superposition", "Drift"]
    s = els[2]
    assert isinstance(s, Superposition) and s.length == pytest.approx((680 + 415.16) * 1e-3)
    assert [z for z, _ in s.children] == pytest.approx([0.0, 0.68])
    names = [n for _, n in s.children]
    assert all(isinstance(lat.elements[n], FieldMap) for n in names)
    assert lat.elements[names[1]].native["tracewin"]["superpose"] == ["680"]
    assert lat.total_length == pytest.approx(0.02 + 1.09516)


def test_single_superpose_map_zero_is_plain_fieldmap(tmp_path):
    deck = "FREQ 352.2\nSUPERPOSE_MAP 0\nFIELD_MAP 100 415.16 153.171 30 0 1.55425 0 0 a\nDRIFT 10 30\nEND\n"
    lat, _ = _read(tmp_path, deck, species="proton")
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Freq", "FieldMap", "Drift"]
    assert els[1].native["tracewin"]["superpose"] == ["0"]


# -- commands / directives ---------------------------------------------------------------------
def test_commands_become_inline_directives(tmp_path):
    deck = (
        "TITLE My Title here\nLATTICE 5 0\nSET_TWISS QUAD1 1.2 0.8 -1 0.6 0 1 1 1 1 1 0 0\n"
        "ADJUST QUAD 2 1 -30 30 0.5 0\n"
        "ERROR_QUAD_NCPL_STAT 1 2 0.1 0.1 0 0 0 0.001 0 0 0 0\nPARTRAN_STEP 100 50\nSPACE_CHARGE_COMP 0.9\n"
        "MIN_EMIT_GROW 1.0 1 0 0 1 1\nRFQ_GEOM 1 2 vane.vane\nMATCH_FAM_GRAD 10 1\nLATTICE_END\n"
        "DRIFT 10 5\nEND\n"
    )
    lat, rep = _read(tmp_path, deck)
    els = _els(lat)
    roles = [(e.card, e.role) for e in els if isinstance(e, Directive)]
    assert roles == [
        ("TITLE", "title"),
        ("LATTICE", "period_start"),
        ("SET_TWISS", "matching"),
        ("ADJUST", "matching"),
        ("ERROR_QUAD_NCPL_STAT", "error"),
        ("PARTRAN_STEP", "tracking"),
        ("SPACE_CHARGE_COMP", "tracking"),
        ("MIN_EMIT_GROW", "matching"),
        ("RFQ_GEOM", "rfq"),
        ("MATCH_FAM_GRAD", "matching"),
        ("LATTICE_END", "period_end"),
    ]
    assert els[1].args == ["5", "0"] and els[0].args == ["My", "Title", "here"]
    assert lat.meta["title"] == "My Title here" and lat.meta["partran_step"] == ["100", "50"]
    assert all(e.length == 0.0 and e.format == "tracewin" for e in els if isinstance(e, Directive))
    assert rep.ok


def test_title_comment_form_is_parsed_back(tmp_path):
    lat, _ = _read(tmp_path, "; TITLE FODO Quadrupole Channel\nDRIFT 10 5\nEND\n")
    d = _first(lat, Directive)
    assert d.role == "title" and lat.meta["title"] == "FODO Quadrupole Channel"


def test_unlisted_command_family_by_prefix(tmp_path):
    lat, rep = _read(tmp_path, "SET_SOMETHING_NEW 1 2\nDRIFT 10 5\nEND\n")
    d = _first(lat, Directive)
    assert d.role == "matching" and d.card == "SET_SOMETHING_NEW"
    assert rep.codes() == {"UNLISTED_COMMAND": 1} and rep.ok


def test_unknown_card_kept_verbatim_dropped(tmp_path):
    lat, rep = _read(tmp_path, "DRIFT 100 20\nUNKNOWN_CARD foo bar\nDRIFT 50 20\nEND\n")
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Drift", "Directive", "Drift"]
    assert els[1].role == "unknown" and els[1].card == "UNKNOWN_CARD" and els[1].args == ["foo", "bar"]
    assert rep.codes() == {"UNKNOWN_CARD": 1} and not rep.ok


def test_set_beam_energy_and_e0_p0(tmp_path):
    deck = (
        "DRIFT 10 5\nSET_BEAM_ENERGY 1 2.5\nDRIFT 10 5\nSET_BEAM_E0_P0 1 0.1 -3.0 1 1\n"
        "SET_BEAM_E0_P0 1 0.1 -3.0 0 1\nEND\n"
    )
    lat, rep = _read(tmp_path, deck)
    rcs = [e for e in _els(lat) if isinstance(e, ReferenceChange)]
    assert rcs[0].energy_eV == pytest.approx(2.5e6) and rcs[0].native["tracewin"]["args"] == ["1", "2.5"]
    assert rcs[1].dE_ref_eV == pytest.approx(0.1e6) and rcs[2].dE_ref_eV is None
    pl = propagate(lat)
    assert pl[2].ref_in.kinetic_energy_eV == pytest.approx(2.5e6)
    assert pl[-1].ref_out.kinetic_energy_eV == pytest.approx(2.6e6)
    assert rep.ok


def test_ncells_length_rule(tmp_path):
    lat, rep = _read(
        tmp_path,
        "FREQ 162.5\nNCELLS 1 4 0.1 5.4e6 -30 15 1\n"
        "NCELLS 0 2 0.2 1e6 -20 15 0 0 0 0 0 0.5 0.8 0.1 0.01\nEND\n",
    )
    a, b = [e for e in _els(lat) if isinstance(e, NCells)]
    lam = C_LIGHT / 162.5e6
    assert a.length == pytest.approx(4 * 0.5 * 0.1 * lam)  # π mode: βλ/2 per cell
    assert b.length == pytest.approx(2 * 0.2 * lam)  # 2π mode: βλ per cell
    assert (
        a.params["mode"] == 1
        and a.params["n_cells"] == 4
        and a.params["p_flag"] == 1
        and a.params["ttf_tail"] == []
    )
    assert b.params["ttf_tail"] == [0.5, 0.8, 0.1, 0.01]
    assert a.rf.voltage_V == pytest.approx(5.4e6 * a.length) and a.rf.phase_rad == pytest.approx(
        math.radians(-30)
    )
    assert rep.codes() == {"RF_ABSOLUTE_PHASE": 1, "SPECIES_ASSUMED": 1} and rep.ok


def test_ncells_beta_g_zero_uses_entrance_beta(tmp_path):
    lat, rep = _read(tmp_path, "FREQ 162.5\nNCELLS 1 4 0 5.4e6 -30 15\nNCELLS 1 4 -0.2 5.4e6 -30 15\nEND\n")
    a, b = [e for e in _els(lat) if isinstance(e, NCells)]
    lam = C_LIGHT / 162.5e6
    assert a.length == pytest.approx(4 * 0.5 * PROTON.beta * lam)
    assert b.length == pytest.approx(4 * 0.5 * 0.2 * lam)
    assert rep.codes()["NCELLS_LENGTH_ESTIMATED"] == 2


def test_rfq_cell(tmp_path):
    lat, rep = _read(
        tmp_path,
        "FREQ 162.5\nRFQ_CELL 80000 3.5 1.2 1.5 12.5 -30 0\n"
        "RFQ_CELL 80000 3.5 1.2 1.5 12.5 -30 2 1 0.5\nEND\n",
    )
    a, b = [e for e in _els(lat) if isinstance(e, RFQCell)]
    assert a.length == pytest.approx(0.0125) and a.rf.voltage_V == 80000 and a.rf.phase_is_sync is True
    assert a.params["cell_type"] == 0 and b.params["tc"] == 1.0 and b.params["dp"] == 0.5
    assert a.aperture.half_x == pytest.approx(0.0035)
    assert rep.ok


def test_helix_comment_cards(tmp_path):
    deck = ";@LG dc_kernel = fast\n; HELIX_FOIL F1 C 100 landau\n; HELIX_SC_GRID 20\nDRIFT 10 5\nEND\n"
    lat, rep = _read(tmp_path, deck)
    els = _els(lat)
    assert isinstance(els[0], Directive) and els[0].card == "@LG" and els[0].args == ["dc_kernel=fast"]
    assert lat.meta["lg_options"] == {"dc_kernel": "fast"}
    f = els[1]
    assert isinstance(f, Foil) and f.name == "F1" and f.material == "C"
    assert f.thickness_kg_per_m2 == pytest.approx(1e-3) and f.native["tracewin"]["straggling"] == "landau"
    assert isinstance(els[2], Directive) and els[2].card == "HELIX_SC_GRID" and els[2].args == ["20"]
    assert rep.ok


def test_lattix_name_tag_names_next_element(tmp_path):
    lat, _ = _read(tmp_path, '; lattix: name="my drift" type="Drift"\nDRIFT 10 5\nDRIFT 10 5\nEND\n')
    assert [e.name for e in _els(lat)] == ["my drift", "DRIFT_0001"]


def test_variables_and_operand_expressions(tmp_path):
    deck = (
        "variable mad2tw   -4.8825\nvariable qfs06_mad    1.21658\n"
        "QFS06H:QUAD     50.000     qfs06_mad*mad2tw     25.400\nEND\n"
    )
    lat, rep = _read(tmp_path, deck)
    q = _first(lat, Quadrupole)
    assert q.name == "QFS06H" and q.gradient == pytest.approx(1.21658 * -4.8825)
    assert q.expressions["gradient"].text == "qfs06_mad*mad2tw"
    assert lat.variables["mad2tw"].value == -4.8825
    assert [e.card for e in _els(lat) if isinstance(e, Directive)] == ["VARIABLE", "VARIABLE"]
    assert rep.codes() == {"EXPRESSION_EVALUATED": 1} and rep.ok


# -- file-level behaviour ------------------------------------------------------------------------
def test_end_comments_and_latin1(tmp_path):
    p = tmp_path / "deck.dat"
    p.write_bytes(b"; comment \xe9\xe8\nDRIFT 100.0 ; inline\nQUAD 50 3 20 5 ; focusing\nEND\nDRIFT 200\n")
    lat, rep = read(p)
    els = _els(lat)
    assert [type(e).__name__ for e in els] == ["Drift", "Quadrupole"]
    assert els[1].gradient == 3.0 and els[1].skew_rad == pytest.approx(math.radians(5))
    assert rep.ok and rep.source_format == "tracewin"


def test_lowercase_end_and_empty_file(tmp_path):
    lat, _ = _read(tmp_path, "DRIFT 100 15\nEnd\nDRIFT 100 15\n")
    assert len(_els(lat)) == 1
    lat2, rep2 = _read(tmp_path, "")
    assert _els(lat2) == [] and rep2.ok and lat2.total_length == 0.0


def test_reference_and_options(tmp_path):
    lat, _ = _read(tmp_path, "FREQ 162.5\nDRIFT 10 5\nEND\n", species="h-", kinetic_energy_eV=800e6)
    assert lat.reference.species.name == "h-" and lat.reference.kinetic_energy_eV == 800e6
    assert lat.reference.rf_frequency_Hz == pytest.approx(162.5e6)
    assert lat.meta["source"].endswith("deck.dat") and lat.name == "deck"
    lat2, _ = Reader().read(tmp_path / "deck.dat", kinetic_energy_eV="3e6", name="mine")
    assert lat2.reference.kinetic_energy_eV == 3e6 and lat2.name == "mine" and lat2.use == "mine"


def test_malformed_card_kept_verbatim(tmp_path):
    lat, rep = _read(tmp_path, "QUAD 100\nDRIFT 10 5\nEND\n")
    d = _first(lat, Directive)
    assert d.role == "unknown" and d.card == "QUAD" and d.args == ["100"]
    assert rep.codes() == {"MALFORMED_CARD": 1}


def test_dtl_deck_exit_energy(tmp_path):
    lat, _ = read(PUBLIC / "helix" / "dtl_section.dat", species="proton")
    out = propagate(lat)[-1].ref_out.kinetic_energy_eV
    assert out == pytest.approx(2.1e6 + 8 * 577000 * math.cos(math.radians(30)))
    assert lat.meta["title"] == "DTL Section (3-7 MeV)"


@pytest.mark.parametrize("deck", sorted(p.relative_to(PUBLIC).as_posix() for p in PUBLIC.rglob("*.dat")))
def test_public_decks_read(deck):
    lat, rep = read(PUBLIC / deck)
    assert len(lat.flatten()) > 0
    assert lat.total_length > 0
    assert not any(e.code in ("UNKNOWN_CARD", "MALFORMED_CARD") for e in rep.entries), rep.summary()


def test_beam_from_the_helix_project_file_next_to_the_deck(tmp_path):
    import json

    (tmp_path / "btl.dat").write_text("DRIFT 100 30\nQUAD 200 5 30\nEND\n")
    (tmp_path / "btl.lgproj").write_text(json.dumps({"__kind__": "helix_project", "lattice_path": "btl.dat",
                                                     "beam": {"species": "H-", "energy": 800.0, "frequency": 162.5}}))
    lat, rep = read(tmp_path / "btl.dat")
    assert lat.reference.species.name == "h-" and lat.reference.kinetic_energy_eV == 800e6
    assert lat.reference.rf_frequency_Hz == 162.5e6
    assert rep.codes().get("REFERENCE_FROM_PROJECT") == 1 and "BEAM_ASSUMED" not in rep.codes()
    assert not lat.warnings
    lat2, rep2 = read(tmp_path / "btl.dat", kinetic_energy_eV=2.0e6)        # an explicit option wins
    assert lat2.reference.kinetic_energy_eV == 2.0e6 and lat2.reference.species.name == "h-"
    assert "BEAM_ASSUMED" not in rep2.codes()


def test_assumed_beam_is_flagged(tmp_path):
    (tmp_path / "bare.dat").write_text("DRIFT 100 30\nEND\n")
    lat, rep = read(tmp_path / "bare.dat")
    assert lat.reference.kinetic_energy_eV == 2.1e6 and lat.reference.species.name == "proton"
    assert rep.codes().get("BEAM_ASSUMED") == 1
    assert lat.warnings and "2.1 MeV" in lat.warnings[0] and "proton" in lat.warnings[0]
    lat, rep = _read(tmp_path, "DRIFT 100 30\nEND\n", kinetic_energy_eV=800e6, species="h-")
    assert "BEAM_ASSUMED" not in rep.codes() and not lat.warnings


def test_trailing_comments_name_cards_and_make_markers(tmp_path):
    """A ``; s NAME TYPE`` comment (decks converted from MAD flat files) or a bare ``; NAME`` names the
    card; a zero-length drift so named is a survey marker; the comment travels in ``meta`` and its type
    words in ``meta["tags"]``; prose comments are kept but name nothing."""
    lat, rep = _read(tmp_path,
                     "DRIFT 300 25.4\n"
                     "DRIFT 0.000 25.400 ; 4.898 HKV MONITOR\n"
                     "DRIFT 0.000 25.400 ; 4.998 HKV MONITOR\n"
                     "DRIFT 200 25.4 ; drift to the buncher\n"
                     "DRIFT 0 25.4\n"
                     "QF1:QUAD 200 5.0 25.4 ; 10.295 QF1 QUADRUPOLE K1=0.5\n"
                     "BEND 10 2000 0 25.4 0 ; 2.450 BA1011 RBEND\n"
                     "MARKER ; 12.5 DCH01 HKICKER\n"
                     "DRIFT 0.000 25.400 ; BLM3\n"
                     "DRIFT 0.000 25.400 0 0 1.0 0 ; 13.0 SHIFTED MONITOR\n")
    els = [lat.elements[it.ref] for it in lat.lines["main"].items] if "main" in lat.lines else \
          [lat.elements[it.ref] for it in next(iter(lat.lines.values())).items]
    names = [e.name for e in els]
    kinds = [e.kind for e in els]
    assert names == ["DRIFT_0001", "HKV", "HKV_2", "DRIFT_0002", "DRIFT_0003", "QF1", "BA1011", "DCH01", "BLM3",
                     "SHIFTED"]
    assert kinds == ["Drift", "Marker", "Marker", "Drift", "Drift", "Quadrupole", "Bend", "Marker", "Marker", "Drift"]
    hkv = els[1]
    assert hkv.meta["comment"] == "4.898 HKV MONITOR" and hkv.meta["tags"] == ["MONITOR"]
    assert hkv.aperture is not None and hkv.native["tracewin"] == {"card": "DRIFT", "args": ["0.000", "25.400"]}
    assert hkv.provenance.original_name is None and hkv.provenance.original_type == "DRIFT"
    assert els[3].meta["comment"] == "drift to the buncher" and "tags" not in els[3].meta
    assert els[5].provenance.original_name == "QF1" and els[5].meta["tags"] == ["QUADRUPOLE", "K1=0.5"]
    assert els[6].meta["tags"] == ["RBEND"] and els[7].meta["tags"] == ["HKICKER"]
    assert "comment" not in els[0].meta and "comment" not in els[4].meta
    assert els[9].kind == "Drift" and els[9].shift is not None            # a shifted zero-length drift is not a marker
