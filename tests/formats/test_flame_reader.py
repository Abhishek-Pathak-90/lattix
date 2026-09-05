"""FLAME GLPS reader tests (PLAN §6 task 3.4).

Three layers:

* the **grammar** — the tokenizer and statement parser are a hand port of FLAME's
  ``src/glps.l``/``src/glps.y``, so every quirk that file has is pinned here
  (identifiers may contain ``:``, numbers may not start with ``.``, ``LINE`` must
  literally be spelled ``LINE``, ``END`` is the only bare command, ``print`` the
  only global function, a trailing comma is legal in a list);
* the **element table** — one hand-computed assertion per conversion rule of
  ``src/moment.cpp`` (``B2`` [T/m] → ``Bn[1]``, ``phi`` [deg] → angle, normalized
  ``K`` [1/m²] → gradient, ``theta_x`` [rad] → ``hkick``, misalignments in m/rad);
* the **corpus** — every vendored deck under ``tests/data/public/flame`` parses,
  with the element counts pinned.

``tests/oracles/test_flame_adapter.py`` runs the engine itself; nothing here needs
FLAME to be installed.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from lattix.fidelity import TranslationError
from lattix.formats.flame import Reader
from lattix.formats.flame.reader import (
    GLPSError,
    charge_and_mass_number,
    parse_statements,
    parse_tags,
    tokenize,
)
from lattix.ir.elements import (
    Bend,
    Drift,
    Foil,
    Instrument,
    Kicker,
    Marker,
    Quadrupole,
    RFCavity,
    Sextupole,
    Solenoid,
    Taylor,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "public" / "flame"

#: element counts of the vendored decks, measured 2026-09-03 (definitions, not placements)
DECKS = {
    "ALL_lattice.lat": {"defs": 2549, "flat": 2549, "length": 471.604917},
    "FrontEnd.lat": {"defs": 138, "flat": 41, "length": 6.05},
    "LS1.lat": {"defs": 1261, "flat": 1261, "length": 158.093655},
    "LS1FS1_lattice.lat": {"defs": 1261, "flat": 1261, "length": 158.093655},
    "TMtest.lat": {"defs": 8, "flat": 8, "length": 0.810386},
    "parse1.lat": {"defs": 1, "flat": 2, "length": 0.0},
}

HEADER = """sim_type = "MomentMatrix";
IonEs = 931494320.0;
IonEk = 500000.0;
IonZ = 33.0/238.0;
IonChargeStates = [33.0/238.0];
NCharge = [1.0];
S: source, vector_variable = "BC", matrix_variable = "S";
"""


def read_text(body: str, header: str = HEADER, **kw):
    return Reader().read_text(header + body, **kw)


def one(body: str, **kw):
    """Parse a one-element snippet and return (element, report)."""
    lat, rep = read_text(body, **kw)
    els = [e for n, e in lat.elements.items() if n != "S"]
    assert len(els) == 1, [e.name for e in els]
    return els[0], rep


# ----------------------------------------------------------------- the lexer
def test_identifier_may_contain_a_colon():
    """``src/glps.l``: ``[A-Za-z]([A-Za-z0-9_:]*[A-Za-z0-9_])?`` — an EPICS PV name is
    ONE token, so ``a:b`` is not ``a`` ``:`` ``b`` and needs the space FLAME decks use."""
    kinds = [(t.kind, t.text) for t in tokenize("LS1:BPM_D1129 x")][:-1]
    assert kinds == [("name", "LS1:BPM_D1129"), ("name", "x")]
    # a trailing colon cannot end an identifier, so `USE: cell;` still parses
    assert [t.text for t in tokenize("USE: cell;")][:-1] == ["USE", ":", "cell", ";"]


def test_numbers_have_no_leading_dot_and_no_sign():
    assert [(t.kind, t.text) for t in tokenize("1 2.5 3. 80.5e6 1e-6")][:-1] == [
        ("num", "1"), ("num", "2.5"), ("num", "3."), ("num", "80.5e6"), ("num", "1e-6")]
    # '.5' is not a number: the '.' is not a GLPS character at all
    with pytest.raises(GLPSError, match="invalid character"):
        tokenize("x = .5;")


def test_comments_and_strings():
    toks = [(t.kind, t.text) for t in tokenize('# hi\na = "some text"; # tail')][:-1]
    assert toks == [("name", "a"), ("punct", "="), ("string", "some text"), ("punct", ";")]


def test_line_numbers_are_tracked():
    toks = tokenize("a = 1;\n\nb = 2;")
    assert [t.line for t in toks if t.kind == "name"] == [1, 3]


# ------------------------------------------------------------- the statements
def test_statement_kinds():
    st = parse_statements('a = 1;\nD: drift, L = 1;\nc: LINE = (D, D);\nUSE: c;\nEND;\nprint(a);')
    assert [s.kind for s in st] == ["assign", "element", "line", "use", "command", "func"]
    assert st[3].type == "c"


def test_line_must_be_spelled_LINE():
    """``glps_parser.cpp:459`` rejects any other keyword in a line-like definition."""
    with pytest.raises(GLPSError, match="instead of 'LINE'"):
        parse_statements("c: SEQ = (a, b);")


def test_only_END_is_a_command_and_only_print_is_a_function():
    read_text("END;\n")
    with pytest.raises(GLPSError, match="undefined command"):
        read_text("STOP;\n")
    read_text("print(1);\n")
    with pytest.raises(GLPSError, match="undefined global function"):
        read_text("show(1);\n")


def test_trailing_comma_is_legal():
    """``LS1.lat:1347`` ships ``(…, drift_61,)``; the grammar's empty ``line_list`` allows it."""
    lat, _ = read_text("D: drift, L = 1;\nc: LINE = (D, D,);\nUSE: c;\n")
    assert len(lat.flatten()) == 2


def test_arithmetic_and_functions():
    lat, _ = read_text("D: drift, L = 2*(1 + 0.5)/3 - cos(0);\nc: LINE = (D);\nUSE: c;\n")
    assert lat.elements["D"].length == pytest.approx(0.0)
    lat, _ = read_text("D: drift, L = deg2rad(180);\nc: LINE = (D);\nUSE: c;\n")
    assert lat.elements["D"].length == pytest.approx(math.pi)


def test_line_repeat_and_reverse():
    """``glps_ops.cpp:236-238``: ``n*line`` repeats and ``-line`` reverses."""
    lat, _ = read_text("A: drift, L = 1;\nB: drift, L = 2;\n"
                       "sub: LINE = (A, B);\nc: LINE = (2*sub, -sub);\nUSE: c;\n")
    flat = [p.element.name for p in lat.flatten()]
    assert flat == ["A", "B", "A", "B", "B", "A"]


def test_nested_lines_are_kept_as_IR_lines():
    """FLAME splices nested lines at parse time; the IR keeps the hierarchy."""
    lat, _ = read_text("A: drift, L = 1;\nsub: LINE = (A, A);\nc: LINE = (sub, A);\nUSE: c;\n")
    assert set(lat.lines) == {"sub", "c"}
    assert [i.ref for i in lat.lines["c"].items] == ["sub", "A"]
    assert len(lat.flatten()) == 3


def test_duplicate_names_are_rejected_like_flame():
    with pytest.raises(GLPSError, match="already used"):
        read_text("A: drift, L = 1;\nA: drift, L = 2;\n")


def test_use_of_an_unknown_line_raises():
    with pytest.raises(GLPSError, match="not a defined line"):
        read_text("A: drift, L = 1;\nUSE: nope;\n")


# ------------------------------------------------------- the reference particle
def test_charge_and_mass_number_from_the_ratio():
    assert charge_and_mass_number(33.0 / 238.0)[:2] == (33, 238)
    assert charge_and_mass_number(1.0)[:2] == (1, 1)
    assert charge_and_mass_number(78.0 / 238.0)[:2] == (39, 119)   # 78/238 reduces


def test_reference_is_per_nucleon():
    """FLAME's IonEs/IonEk are eV/u and IonZ = Q/A (``src/flame/moment.h:57``)."""
    lat, _ = read_text("D: drift, L = 1;\nc: LINE = (D);\nUSE: c;\n")
    ref = lat.reference
    assert ref.species.charge == 33
    assert ref.species.mass_eV == pytest.approx(931494320.0 * 238)
    assert ref.kinetic_energy_eV == pytest.approx(500000.0 * 238)
    # gamma is per-nucleon-invariant, and Brho matches FLAME's beta*IonW/(C0*IonZ)
    assert ref.gamma == pytest.approx(1 + 500000.0 / 931494320.0)
    flame_brho = ref.beta * (931494320.0 + 500000.0) / (299792458.0 * (33.0 / 238.0))
    assert ref.brho_abs == pytest.approx(flame_brho, rel=1e-14)
    assert lat.meta["flame"]["mass_number"] == 238


def test_multiple_charge_states_keep_the_first():
    header = HEADER.replace("IonChargeStates = [33.0/238.0];",
                            "IonChargeStates = [33.0/238.0, 34.0/238.0];")
    lat, rep = read_text("D: drift, L = 1;\nc: LINE = (D);\nUSE: c;\n", header)
    assert lat.reference.species.charge == 33
    assert "MULTI_CHARGE_STATE_FIRST" in rep.codes()


def test_sample_freq_is_the_reference_clock():
    lat, _ = read_text("D: drift, L = 1;\nc: LINE = (D);\nUSE: c;\n")
    assert lat.reference.rf_frequency_Hz == 80.5e6          # SampleFreqDefault
    lat, _ = read_text("D: drift, L = 1;\nc: LINE = (D);\nUSE: c;\n",
                       HEADER + "SampleFreq = 162.5e6;\n")
    assert lat.reference.rf_frequency_Hz == 162.5e6


# ---------------------------------------------------------- the element table
def test_drift():
    el, _ = one("D: drift, L = 0.25, aper = 0.02;\n")
    assert isinstance(el, Drift) and el.length == 0.25
    assert el.aperture.half_x == pytest.approx(0.02)        # aper is a radius [m]


def test_quadrupole_B2_is_a_lab_gradient():
    el, _ = one("Q: quadrupole, L = 0.3, B2 = -12.2835;\n")
    assert isinstance(el, Quadrupole)
    assert el.multipole.Bn[1] == -12.2835                   # T/m, straight through


def test_sextupole_B3():
    el, _ = one("SX: sextupole, L = 0.2, B3 = 3.5;\n")
    assert isinstance(el, Sextupole) and el.multipole.Bn[2] == 3.5


def test_solenoid_B():
    el, _ = one("SOL: solenoid, L = 0.1, B = 5.34;\n")
    assert isinstance(el, Solenoid) and el.solenoid.Bsol_T == 5.34


def test_sbend_angles_are_degrees_and_K_is_normalized():
    """``moment.cpp:1015-1020``: phi/phi1/phi2 in deg, ``K`` read as ``K/sqr(MtoMM)``
    i.e. 1/m² — and ``moment_sup.cpp:228`` uses it as MAD-X's ``k1`` (Kx = K + 1/ρ²)."""
    el, _ = one("B: sbend, L = 1.0, phi = 5.0, phi1 = 2.5, phi2 = 2.5, K = 0.4, bg = 0.5;\n")
    assert isinstance(el, Bend)
    assert el.bend.angle == pytest.approx(math.radians(5.0))
    assert el.bend.e1 == pytest.approx(math.radians(2.5))
    assert el.bend.e2 == pytest.approx(math.radians(2.5))
    lat, _ = read_text("B: sbend, L = 1.0, phi = 5.0, phi1 = 0, phi2 = 0, K = 0.4;\n"
                       "c: LINE = (S, B);\nUSE: c;\n")
    ref = lat.reference
    assert lat.elements["B"].multipole.Bn[1] == pytest.approx(0.4 * ref.brho_signed)
    assert lat.elements["B"].meta["flame_K"] == 0.4
    assert el.meta["flame_bg"] == 0.5


def test_sbend_ver_is_a_vertical_bend():
    el, _ = one("B: sbend, L = 1.0, phi = 5.0, phi1 = 0, phi2 = 0, ver = 1;\n")
    assert el.bend.tilt_ref == pytest.approx(math.pi / 2)


def test_orbtrim_theta_is_a_deflection_in_rad():
    """``moment.cpp:897``: ``transfer(PS_PX, 6) = +theta_x`` — the MAD-X hkick sign."""
    el, _ = one("K: orbtrim, theta_x = 0.001, theta_y = -0.002;\n")
    assert isinstance(el, Kicker)
    assert (el.hkick, el.vkick) == (0.001, -0.002)


def test_orbtrim_realpara_uses_the_integrated_field():
    lat, _ = read_text("K: orbtrim, realpara = 1, tm_xkick = 0.01, tm_ykick = 0;\n"
                       "c: LINE = (S, K);\nUSE: c;\n")
    k = lat.elements["K"]
    assert k.hkick == pytest.approx(0.01 / lat.reference.brho_signed)


def test_rfcavity_keeps_cavtype_frequency_and_phase():
    el, rep = one('C: rfcavity, cavtype = "0.041QWR", L = 0.24, f = 80.5e6, '
                  "phi = -35.0, scl_fac = 0.64, aper = 0.017;\n")
    assert isinstance(el, RFCavity)
    assert el.rf.frequency_Hz == 80.5e6
    assert el.rf.phase_rad == pytest.approx(math.radians(-35.0))
    assert el.rf.phase_is_sync is True                     # syncflag defaults to 1
    assert el.rf.voltage_V == 0.0                          # not in the deck
    assert el.meta["flame_cavtype"] == "0.041QWR"
    assert el.meta["flame_scl_fac"] == 0.64
    assert "FLAME_CAVTYPE_VOLTAGE_UNKNOWN" in rep.codes()


def test_rfcavity_syncflag_zero_is_a_driven_phase():
    el, rep = one('C: rfcavity, cavtype = "Generic", L = 0.24, f = 80.5e6, phi = -35.0, '
                  "syncflag = 0;\n")
    assert el.rf.phase_is_sync is False
    assert "FLAME_DRIVEN_PHASE" in rep.codes()


def test_markers_bpms_and_strippers():
    assert isinstance(one("M: marker;\n")[0], Marker)
    bpm = one("P: bpm;\n")[0]
    assert isinstance(bpm, Instrument) and bpm.family == "BPM"
    foil, rep = one("ST: stripper, IonChargeStates = [0.3], NCharge = [1.0];\n")
    assert isinstance(foil, Foil)
    assert "FLAME_STRIPPER_MODEL" in rep.codes()


def test_tmatrix_splits_the_7x7_into_map_and_offset():
    body = "TM: tmatrix, matrix = [" + ", ".join(
        "1" if i == j else ("3" if (i, j) == (2, 6) else "0")
        for i in range(7) for j in range(7)) + "];\n"
    el, _ = one(body)
    assert isinstance(el, Taylor)
    assert el.matrix[2][2] == 1.0
    assert el.offset == [0.0, 0.0, 3.0e-3, 0.0, 0.0, 0.0]  # the 7th column is the kick; FLAME's y is in mm
    assert el.basis == "common"                              # transverse block in the IR's SI units
    assert el.meta["flame_matrix_row6"][6] == 1.0


def test_electrostatic_elements_are_dropped_but_keep_their_length():
    el, rep = one("EQ: equad, L = 0.1, V = 5000, radius = 0.075;\n")
    assert isinstance(el, Marker) and el.length == 0.1
    assert "ELECTROSTATIC_UNSUPPORTED" in rep.codes()
    assert el.native["flame"]["type"] == "equad"
    assert el.native["flame"]["attrs"]["V"] == 5000


def test_misalignments_are_metres_and_radians():
    el, _ = one("Q: quadrupole, L = 0.3, B2 = 1.0, dx = 1e-4, dy = -2e-4, "
                "pitch = 1e-3, yaw = 2e-3, roll = 3e-3;\n")
    s = el.shift
    assert (s.x_offset, s.y_offset) == (1e-4, -2e-4)
    assert (s.x_rot, s.y_rot, s.tilt) == (1e-3, 2e-3, 3e-3)


def test_unknown_type_is_a_marker_in_permissive_mode():
    el, rep = one("X: nosuchtype, L = 1;\n")
    assert isinstance(el, Marker)
    assert "FLAME_UNKNOWN_TYPE" in rep.codes()


# ------------------------------------------------------------- dual regime
def test_strict_raises_on_a_dropped_element():
    with pytest.raises(TranslationError, match="ELECTROSTATIC_UNSUPPORTED"):
        read_text("EQ: equad, L = 0.1, V = 5000, radius = 0.075;\n"
                  "c: LINE = (S, EQ);\nUSE: c;\n", strict=True)


def test_strict_raises_on_a_bad_expression():
    with pytest.raises(TranslationError, match="FLAME_BAD_EXPRESSION"):
        read_text("D: drift, L = nosuchvar;\nc: LINE = (S, D);\nUSE: c;\n", strict=True)


def test_permissive_drops_a_bad_attribute_and_keeps_reading():
    lat, rep = read_text("D: drift, L = 1, aper = nosuchvar;\nc: LINE = (S, D);\nUSE: c;\n")
    assert lat.elements["D"].length == 1.0
    assert lat.elements["D"].aperture is None
    assert "FLAME_BAD_EXPRESSION" in rep.codes()


# ------------------------------------------------------------------ the corpus
@pytest.mark.parametrize("name", sorted(DECKS))
def test_vendored_deck_parses(name):
    expected = DECKS[name]
    lat, rep = Reader().read(DATA / name)
    placed = lat.flatten()
    assert len(lat.elements) == expected["defs"]
    assert len(placed) == expected["flat"]
    assert sum(p.length for p in placed) == pytest.approx(expected["length"], abs=1e-9)
    assert isinstance(rep.codes(), dict)


def test_ALL_lattice_covers_the_element_types():
    """``ALL_lattice.lat`` is FRIB's every-element deck; these are the codes assigned."""
    lat, _ = Reader().read(DATA / "ALL_lattice.lat")
    counts: dict[str, int] = {}
    for el in lat.elements.values():
        t = el.native["flame"]["type"]
        counts[t] = counts.get(t, 0) + 1
    assert counts == {"source": 1, "bpm": 33, "drift": 1183, "orbtrim": 200, "rfcavity": 332,
                      "solenoid": 276, "quadrupole": 103, "marker": 220, "sbend": 200,
                      "stripper": 1}
    kinds: dict[str, int] = {}
    for el in lat.elements.values():
        kinds[el.kind] = kinds.get(el.kind, 0) + 1
    assert kinds == {"Marker": 221, "Instrument": 33, "Drift": 1183, "Kicker": 200,
                     "RFCavity": 332, "Solenoid": 276, "Quadrupole": 103, "Bend": 200,
                     "Foil": 1}


def test_front_end_has_the_electrostatic_elements_and_a_deck_bug():
    """FrontEnd.lat is the only vendored deck with ``edipole``/``equad`` — and it ships
    ``ver = v`` with ``v`` undefined, which FLAME itself rejects."""
    lat, rep = Reader().read(DATA / "FrontEnd.lat")
    types = [el.native["flame"]["type"] for el in lat.elements.values()]
    assert types.count("edipole") == 2
    assert types.count("equad") == 8
    codes = rep.codes()
    assert codes["ELECTROSTATIC_UNSUPPORTED"] == 10
    assert codes["FLAME_BAD_EXPRESSION"] == 2
    assert any("undefined variable 'v'" in w for w in lat.warnings)


def test_tmtest_has_the_only_tmatrix():
    lat, _ = Reader().read(DATA / "TMtest.lat")
    tm = [e for e in lat.elements.values() if isinstance(e, Taylor)]
    assert len(tm) == 1
    # x, y (mm in FLAME) come back in metres; the phase offset stays in FLAME's rad
    assert tm[0].offset[0] == 1.0e-3 and tm[0].offset[2] == 3.0e-3 and tm[0].offset[4] == 5.0


def test_globals_are_kept_in_meta():
    lat, _ = Reader().read(DATA / "LS1FS1_lattice.lat")
    g = lat.meta["flame"]
    assert g["globals"]["sim_type"] == "MomentMatrix"
    assert g["globals"]["MpoleLevel"] == "2"
    assert g["global_exprs"]["IonZ"] == "(33 / 238)"
    assert len(g["beam_vectors"]["BaryCenter0"]) == 7
    assert len(g["beam_vectors"]["S0"]) == 49


# ------------------------------------------------------------------ name tags
def test_parse_tags_reads_the_writer_comment():
    text = '# lattix: name="a b" type="edipole"\nX: marker;\n'
    assert parse_tags(text) == {2: ("a b", "edipole")}
    lat, _ = read_text(text)
    assert lat.elements["X"].provenance.original_name == "a b"
    assert lat.elements["X"].provenance.original_type == "edipole"
