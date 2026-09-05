"""Fidelity honesty for the TraceWin reader and writer: every LOSSY/DROPPED path records its code in
permissive mode and raises ``TranslationError`` in strict mode; every EQUIVALENT path is recorded
but never fatal (pattern of HELIX tests/io/test_parser_downgrades.py, both regimes per card)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix.fidelity import FidelityClass, TranslationError
from lattix.formats.tracewin import Writer, read, render, write
from lattix.ir import (
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
    Instrument,
    Kicker,
    Lattice,
    MagneticMultipoleP,
    Multipole,
    Octupole,
    Patch,
    Quadrupole,
    ReferenceParticle,
    RFCavity,
    Sextupole,
    Superposition,
    Taylor,
    species,
)
from lattix.ir.reference_tag import format_reference_tag

PROTON = ReferenceParticle(species=species("proton"), kinetic_energy_eV=2.1e6, rf_frequency_Hz=162.5e6)


def _parse(tmp_path, text, **kw):
    p = tmp_path / "deck.dat"
    p.write_text(text)
    return read(p, **kw)


def _lat(*elements):
    return Lattice.from_sequence("l", list(elements), PROTON)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("1\n0 1\n0\n1\n")


# ── reader: LOSSY / DROPPED cards — permissive records, strict raises ─────────────────────────
_READER_PROBLEMS = [
    ("UNKNOWN_CARD", "FOOBAR 1 2 3", FidelityClass.DROPPED),
    ("MALFORMED_CARD", "QUAD 100", FidelityClass.DROPPED),
    ("MALFORMED_CARD", "DRIFT abc 15", FidelityClass.DROPPED),
    ("ORPHAN_EDGE", "EDGE 5 1000", FidelityClass.LOSSY),
    ("FM_FILES_MISSING", "FREQ 162.5\nFIELD_MAP 100 200 0 15 0 1.0 0 0 nofile", FidelityClass.LOSSY),
    ("FM_GEOM_UNSUPPORTED", "FREQ 162.5\nFIELD_MAP 8 200 0 15 0 1.0 0 0 nofile", FidelityClass.LOSSY),
    ("APERTURE_SHAPE", "APERTURE 10 9 5", FidelityClass.LOSSY),
    ("GAP_ABSOLUTE_PHASE", "FREQ 162.5\nGAP 1e6 -30 15 1", FidelityClass.LOSSY),
    ("SUPERPOSE_MAP_OUT_UNSUPPORTED", "SUPERPOSE_MAP_OUT 100 0 0 0 0 0", FidelityClass.LOSSY),
    ("BEAM_DIRECTIVE", "READ_DST beam.dst", FidelityClass.LOSSY),
    ("BEAM_DIRECTIVE", "BEAM_ROT 0 0 30", FidelityClass.LOSSY),
    ("REPEAT_ELE_NOT_EXPANDED", "REPEAT_ELE 10 1\nQUAD 10 15 30", FidelityClass.LOSSY),
    ("SUPERPOSE_DANGLING", "SUPERPOSE_MAP 0", FidelityClass.LOSSY),
    (
        "SUPERPOSE_DANGLING",
        "SUPERPOSE_MAP 0\nSUPERPOSE_MAP 10\nFIELD_MAP 100 200 0 15 0 1.0 0 0 nofile",
        FidelityClass.LOSSY,
    ),
    ("UNSUPPORTED_ELEMENT", "THIN_LENS 1000 1000 15", FidelityClass.LOSSY),
    ("UNSUPPORTED_ELEMENT", "DTL_CEL 100 10 10 50 -50 1e6 -30 10 0", FidelityClass.LOSSY),
]


@pytest.mark.parametrize(
    "code,card,cls",
    _READER_PROBLEMS,
    ids=[f"{c}:{k.split(chr(10))[-1].split()[0]}" for c, k, _ in _READER_PROBLEMS],
)
def test_reader_downgrade_permissive_records_strict_raises(tmp_path, code, card, cls):
    deck = f"DRIFT 100 15 0\n{card}\nDRIFT 100 15 0\nEND\n"
    lat, rep = _parse(tmp_path, deck, species="proton")
    hits = [e for e in rep.entries if e.code == code]
    assert hits, rep.summary()
    assert all(e.cls is cls for e in hits)
    assert all(e.line is not None for e in hits)
    assert not rep.ok
    # the deck is still fully represented: both drifts are there and the card survived verbatim
    assert sum(isinstance(p.element, Drift) for p in lat.flatten()) == 2
    with pytest.raises(TranslationError) as ei:
        _parse(tmp_path, deck, strict=True, species="proton")
    assert ei.value.entry.cls in (FidelityClass.LOSSY, FidelityClass.DROPPED)


def test_cluster_split_by_freq(tmp_path):
    _touch(tmp_path / "a.edz")
    deck = (
        "FREQ 162.5\nSUPERPOSE_MAP 0\nFIELD_MAP 100 100 0 15 0 1 0 0 a\nFREQ 200\nSUPERPOSE_MAP 50\n"
        "FIELD_MAP 100 100 0 15 0 1 0 0 a\nDRIFT 10 5\nEND\n"
    )
    lat, rep = _parse(tmp_path, deck, species="proton")
    assert rep.codes() == {"CLUSTER_SPLIT_BY_FREQ": 1}
    kinds = [p.element.kind for p in lat.flatten()]
    # laid out sequentially: the lone z0=0 map is plain, the z0=50 map keeps its offset container
    assert kinds == ["Freq", "FieldMap", "Freq", "Superposition", "Drift"]
    assert lat.total_length == pytest.approx(0.1 + 0.15 + 0.01)
    with pytest.raises(TranslationError, match="CLUSTER_SPLIT_BY_FREQ"):
        _parse(tmp_path, deck, strict=True, species="proton")


def test_missing_field_file_element_is_kept_not_dropped(tmp_path):
    """Unlike HELIX (permissive DROP), the cavity is kept so downstream s / energies keep their slots."""
    deck = (
        "FIELD_MAP_PATH .\nDRIFT 100 15 0\nFREQ 162.5\nFIELD_MAP 100 200 0 15 0 1.0 0 0 nofile\n"
        "DRIFT 100 15 0\nEND\n"
    )
    lat, rep = _parse(tmp_path, deck, species="proton")
    assert [p.element.kind for p in lat.flatten()] == ["Drift", "Freq", "FieldMap", "Drift"]
    assert lat.total_length == pytest.approx(0.4)
    assert rep.codes() == {"FM_FILES_MISSING": 1}
    with pytest.raises(TranslationError, match="FM_FILES_MISSING"):
        _parse(tmp_path, deck, strict=True, species="proton")


# ── reader: EQUIVALENT paths — recorded, never strict-fatal ────────────────────────────────────
_READER_EQUIVALENT = [
    ("EXTRA_TOKENS_IGNORED", "SOLENOID 100 0.3 25 20"),
    ("FREQ_ASSUMED", "GAP 1e6 -30 15"),
    ("SHIFT_IN_FIELD_MAP_INLINE", "SHIFT_IN_FIELD_MAP 5\nDIAG_SIZE 1"),
    ("RF_ABSOLUTE_PHASE", "FREQ 162.5\nNCELLS 1 4 0.1 5.4e6 -30 15 1"),
    ("RF_ABSOLUTE_PHASE", "FREQ 162.5\nFIELD_MAP 100 200 0 15 0 1.0 0 0 a 1"),
    ("SUPERPOSE_MAP_OFFSETS_IGNORED", "FREQ 162.5\nSUPERPOSE_MAP 0 1 0\nFIELD_MAP 100 200 0 15 0 1.0 0 0 a"),
    ("UNLISTED_COMMAND", "SET_FANCY_NEW 1 2 3"),
    ("EXPRESSION_EVALUATED", "VARIABLE g 2.5\nQUAD 100 g*2 15"),
    ("NCELLS_LENGTH_ESTIMATED", "FREQ 162.5\nNCELLS 1 4 0 5.4e6 -30 15"),
]


@pytest.mark.parametrize("code,card", _READER_EQUIVALENT, ids=[c for c, _ in _READER_EQUIVALENT])
def test_reader_equivalent_recorded_not_fatal(tmp_path, code, card):
    _touch(tmp_path / "a.edz")
    deck = f"DRIFT 100 15 0\n{card}\nDRIFT 100 15 0\nEND\n"
    lat, rep = _parse(tmp_path, deck, species="proton")
    assert code in rep.codes(), rep.summary()
    assert all(e.cls is FidelityClass.EQUIVALENT for e in rep.entries if e.code == code)
    assert rep.ok
    _parse(tmp_path, deck, strict=True, species="proton")  # strict still parses


def test_species_assumed_only_when_defaulted(tmp_path):
    deck = "FREQ 162.5\nGAP 1e6 -30 15\nEND\n"
    _, rep = _parse(tmp_path, deck)
    assert rep.codes() == {"SPECIES_ASSUMED": 1} and rep.ok
    _, rep_given = _parse(tmp_path, deck, species="proton")
    assert "SPECIES_ASSUMED" not in rep_given.codes()
    _, rep_sync = _parse(tmp_path, "FREQ 162.5\nSET_SYNC_PHASE\nGAP 1e6 -30 15\nEND\n")
    assert "SPECIES_ASSUMED" not in rep_sync.codes()  # a synchronous phase is species independent
    _parse(tmp_path, deck, strict=True)


def test_clean_deck_has_no_entries_but_exact(tmp_path):
    _, rep = _parse(tmp_path, "FREQ 162.5\nDRIFT 100 15\nQUAD 50 5 15\nEND\n", strict=True)
    assert rep.codes() == {} and rep.counts == {"EXACT": 3}


# ── writer: LOSSY / DROPPED kinds — permissive writes with the code, strict raises before writing ──
def _fm(**kw):
    return FieldMap(name="fm", length=0.2, geom=100, files=["a"], rf=RFP(frequency_Hz=162.5e6), **kw)


_WRITER_PROBLEMS = [
    ("FOIL_AS_COMMENT", Foil(name="F1", material="C", thickness_kg_per_m2=1e-3), FidelityClass.LOSSY),
    (
        "THIN_MULTIPOLE_UNSUPPORTED",
        Multipole(name="m", multipole=MagneticMultipoleP(BnL={2: 1.0})),
        FidelityClass.DROPPED,
    ),
    ("TAYLOR_UNSUPPORTED", Taylor(name="t"), FidelityClass.DROPPED),
    ("PATCH_UNSUPPORTED", Patch(name="p", x_offset=1e-3), FidelityClass.DROPPED),
    (
        "FOREIGN_DIRECTIVE",
        Directive(name="d", format="madx", card="EALIGN", args=["dx=1e-3"]),
        FidelityClass.DROPPED,
    ),
    (
        "MISALIGN_DROPPED",
        Quadrupole(name="q", length=0.1, shift=BodyShiftP(x_offset=1e-3)),
        FidelityClass.LOSSY,
    ),
    ("MISALIGN_DROPPED", Drift(name="d", length=0.1, shift=BodyShiftP(tilt=1e-3)), FidelityClass.LOSSY),
    (
        "SKEW_MULTIPOLE_DROPPED",
        Quadrupole(name="q", length=0.1, multipole=MagneticMultipoleP(Bn={1: 2.0}, Bs={1: 0.5})),
        FidelityClass.LOSSY,
    ),
    (
        "BEND_TILT_UNSUPPORTED",
        Bend(name="b", length=0.5, bend=BendP(angle=0.1, tilt_ref=0.3)),
        FidelityClass.LOSSY,
    ),
    (
        "BEND_MULTIPOLE_DROPPED",
        Bend(name="b", length=0.5, bend=BendP(angle=0.1), multipole=MagneticMultipoleP(Bn={2: 3.0})),
        FidelityClass.LOSSY,
    ),
    (
        "APERTURE_SHAPE",
        Drift(
            name="d",
            length=0.1,
            aperture=ApertureP(shape="ELLIPTICAL", x_limits=(-0.02, 0.02), y_limits=(-0.01, 0.01)),
        ),
        FidelityClass.LOSSY,
    ),
    (
        "FIELDMAP_NO_SOURCE",
        FieldMap(name="fm", length=0.2, rf=RFP(frequency_Hz=162.5e6)),
        FidelityClass.DROPPED,
    ),
]


@pytest.mark.parametrize("code,element,cls", _WRITER_PROBLEMS, ids=[c for c, _, _ in _WRITER_PROBLEMS])
def test_writer_downgrade_permissive_records_strict_raises(tmp_path, code, element, cls):
    lat = _lat(Drift(name="d1", length=0.1), element, Drift(name="d2", length=0.1))
    out = tmp_path / "out.dat"
    rep = write(lat, out)
    hits = [e for e in rep.entries if e.code == code]
    assert hits and all(e.cls is cls and e.element == element.name for e in hits), rep.summary()
    assert not rep.ok
    text = out.read_text()
    assert "d1: DRIFT 100" in text and "d2: DRIFT 100" in text and text.endswith("END\n")
    strict_out = tmp_path / "strict.dat"
    with pytest.raises(TranslationError) as ei:
        write(lat, strict_out, strict=True)
    assert ei.value.entry.code == code or ei.value.entry.cls in (FidelityClass.LOSSY, FidelityClass.DROPPED)
    assert not strict_out.exists()  # strict never leaves a half-faithful deck behind


def test_writer_superposition_child_must_be_fieldmap(tmp_path):
    lat = Lattice(name="l", reference=PROTON)
    lat.add_element(Drift(name="child", length=0.1))
    lat.add_element(Superposition(name="s", length=0.1, children=[(0.0, "child")]))
    from lattix.ir import Line, LineItem

    lat.lines["l"] = Line(name="l", items=[LineItem(ref="s")])
    lat.use = "l"
    text, rep = render(lat, tmp_path)
    assert "SUPERPOSE_CHILD_UNSUPPORTED" in rep.codes() and not rep.ok


# ── writer: EQUIVALENT kinds — recorded, never strict-fatal ────────────────────────────────────
_WRITER_EQUIVALENT = [
    (
        "THICK_CAVITY_AS_GAP",
        RFCavity(name="c", length=0.3, rf=RFP(voltage_V=1e6, phase_rad=-0.5, frequency_Hz=162.5e6)),
    ),
    (
        "THICK_MULTIPOLE_AS_QUAD_CARD",
        Sextupole(name="s", length=0.1, multipole=MagneticMultipoleP(Bn={2: 3.0})),
    ),
    (
        "THICK_MULTIPOLE_AS_QUAD_CARD",
        Octupole(name="o", length=0.1, multipole=MagneticMultipoleP(Bn={3: 3.0})),
    ),
    ("ZERO_ANGLE_BEND_AS_DRIFT", Bend(name="b", length=0.5)),
    ("INSTRUMENT_AS_MARKER", Instrument(name="i", family="PROFILE")),
    ("ZERO_MULTIPOLE_OMITTED", Multipole(name="m")),
    ("THICK_KICKER_SPLIT", Kicker(name="k", length=0.1, hkick=1e-3)),
    ("COLLIMATOR_NO_APERTURE", Collimator(name="c")),
]


@pytest.mark.parametrize(
    "code,element", _WRITER_EQUIVALENT, ids=[f"{c}:{e.kind}" for c, e in _WRITER_EQUIVALENT]
)
def test_writer_equivalent_recorded_not_fatal(tmp_path, code, element):
    lat = _lat(Drift(name="d1", length=0.1), element, Drift(name="d2", length=0.1))
    rep = write(lat, tmp_path / "out.dat", strict=True)
    assert code in rep.codes(), rep.summary()
    assert all(e.cls is FidelityClass.EQUIVALENT for e in rep.entries if e.code == code)
    assert rep.ok


def test_writer_sync_phase_promoted(tmp_path):
    raw = RFCavity(
        name="g",
        rf=RFP(voltage_V=1e6, phase_rad=math.radians(-30), frequency_Hz=162.5e6, phase_is_sync=False),
    )
    lat = _lat(Directive(name="s", card="SET_SYNC_PHASE", role="sync_phase"), raw)
    text, rep = render(lat, tmp_path)
    assert (
        text == format_reference_tag(lat.reference, ";") + "\nFREQ 162.5\nSET_SYNC_PHASE\ng: GAP 1000000 -30 0\nEND\n"
    )  # FREQ lands before the directive
    assert rep.codes() == {"SYNC_PHASE_PROMOTED": 1} and rep.ok


def test_writer_one_entry_per_placed_element(tmp_path):
    lat = _lat(
        Drift(name="d", length=0.1),
        Quadrupole(name="q", length=0.1),
        Foil(name="f"),
        RFCavity(name="c", rf=RFP(voltage_V=1e6, frequency_Hz=162.5e6)),
    )
    _, rep = render(lat, tmp_path)
    assert sorted(e.element for e in rep.entries) == ["c", "d", "f", "q"]
    assert Writer().format == "tracewin"
